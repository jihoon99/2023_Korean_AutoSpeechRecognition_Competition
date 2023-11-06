import torch
import numpy as np
import math
from dataclasses import dataclass
from pyctcdecode import build_ctcdecoder
import time
import json
from nova import DATASET_PATH

import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import torch.nn.functional as F
from torch.cuda.amp import (autocast, GradScaler)

import multiprocessing

def testing_pred_lengths(preds):
    preds = F.log_softmax(preds, dim=-1)
    preds_lengths = torch.sum(torch.ones_like(preds[:,:,0]).long(), dim=-1)
    return preds_lengths


def after_decode(str_ls:str):
    return str_ls.replace("<eos>", "")

def training(
        config, 
        dataloader,
        optimizer,
        model, 
        criterion, 
        cer_metric,
        wer_metric,
        train_begin_time, 
        device,
        vocab,
        decoder):

    model.train()

    scaler = GradScaler()
    
    # log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
    #                           "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] TRAINING Start')
    epoch_begin_time = time.time()
    cnt = 0
    cer_ls = []


    for inputs, targets, input_lengths, target_lengths in dataloader: # input_lengths : audio seq length, target_length : token length
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        # print(f'input_lengths : {input_lengths.tolist()}')
        
        with autocast():
            outputs, output_lengths = model(inputs, input_lengths)
            # print(f'output_lengths : {output_lengths.tolist()}')

        # Use AMP
        #with autocast():
        #    for _output, _output_length, _target, _target_length in zip(outputs, output_lengths, targets, target_lengths):
        #        _output = _output.unsqueeze(0)
                
        #        without_pad_target = _target[1:_target_length]
        #        without_pad_target = without_pad_target.unsqueeze(0)

            #loss = criterion(
            #        outputs.transpose(0, 1),
            #        targets[:, 1:],
            #        tuple(_output_length),
            #        tuple(_target_length-1)
            #    )

           #     GradScaler.scale(loss).backward()

            #GradScaler.step(optimizer)
            #GradScaler.update()
            #optimizer.zero_grad()



            input_lengths = testing_pred_lengths(outputs)

            loss = criterion(
                outputs.log_softmax(-1).transpose(0, 1),
                targets[:, 1:],
                tuple(input_lengths),
                tuple(target_lengths-1)
                )


        y_hats = torch.argmax(outputs.log_softmax(-1), dim=-1)
        # batch 128 크다 그러니까 : cumulate backward step 방법론 생각해봄직함. 
        # optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #loss.backward()
        #optimizer.step()

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()


        # if cnt % config.print_every == 0:
        #     print(f'y_hat : {y_hats[0]}')
        #     print(f'targets : {targets[0]}')


        if cnt % config.print_every == 1:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0    # 아 초, 단위로 한거구나.
            train_elapsed = (current_time - train_begin_time) / 3600.0  # 시간 단위로 변환한거구나.
            _targets = targets[:,1:]
            _targets = [after_decode(decoder.decode(t)) for t in _targets] 
            _y_hats = [after_decode(decoder.decode(y_)) for y_ in y_hats]
            cer = cer_metric.compute(references=_targets, predictions=_y_hats)
            cer_ls += [cer]
            #wer = wer_metric(targets[:, 1:], y_hats)
            wer = 0
            print(f'[INFO] TRAINING step : {cnt:4d}/{len(dataloader):4d}, mean_loss : {epoch_loss_total/cnt:.6f}, mean_cer : {np.mean(cer_ls):.2f}, current_cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m {train_elapsed:.2f}h')
            print('-'*100)
            print(_targets[0])
            print(_y_hats[0])
            print('-'*100)

            # print(log_format.format(
            #     cnt, len(dataloader), loss,
            #     cer, elapsed, epoch_elapsed, train_elapsed,
            #     # optimizer.get_lr(),
            # ))
        cnt += 1
        torch.cuda.empty_cache()

    return model, epoch_loss_total/len(dataloader), np.mean(cer_ls)

def decoderWithLM(vocab_fn, lm_fn):
    '''
    vocab_fn : config.vocab_json_fn
    lm_fn : 추가해야함.   5gram_correct.arpa
    '''
    def load_vocab_dict(fn):
        '''
            fn : unit2id.json
        '''
        with open(fn, 'r') as f:
            vocab_dict =json.load(f)
        return vocab_dict
    def build_decoder(lm_fn, vocab_dict):
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()),
            kenlm_model_path=lm_fn,
        )
        return decoder
    vocab_dict = load_vocab_dict(vocab_fn)
    decoder = build_decoder(lm_fn, vocab_dict)
    return decoder

def validating(
        config, 
        dataloader, 
        optimizer, 
        model, 
        criterion, 
        cer_metric, 
        wer_metric,
        train_begin_time, 
        device, 
        vocab,
        decoder):

    print(config.vocab_json_fn, config.lm_fn)
    # @hijung - Add LM -> decoder hard coding
    decoder_with_lm = decoderWithLM(config.vocab_json_fn,
                                    config.lm_fn)
    model.eval()

    # log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
    #                           "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] VALIDATING Start')
    epoch_begin_time = time.time()
    cnt = 0
    cer_ls = []

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in dataloader: # input_lengths : audio seq length, target_length : token length
            begin_time = time.time()

            # optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            # model = model.to(device) # 모델을 불러 올 때 이미 gpu에 올림.

            with autocast():
                outputs, output_lengths = model(inputs, input_lengths)

            ######### 이부분 accumulate으로 변경



                input_lengths = testing_pred_lengths(outputs)

                loss = criterion(
                    outputs.log_softmax(-1).transpose(0, 1),
                    targets[:, 1:],
                    tuple(input_lengths),
                    tuple(target_lengths-1)
                    )


            y_hats = torch.argmax(outputs.log_softmax(-1), dim=-1)

            # if mode == 'train':
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            total_num += int(input_lengths.sum())
            epoch_loss_total += loss.item()


            # if cnt % int(config.print_every) == 0:
            #     for i in range(3):
            #         print(f'y_hat : {y_hats[0]}')
                    # print(f'targets : {targets[0]}')
                    # print(f'y_hat_decoded : {y_hat_decoded[i]}')
                    # print(f'targets_decoded : {targets_decoded[i]}')


            if cnt % config.print_every == 1:
                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0
                train_elapsed = (current_time - train_begin_time) / 3600.0

                _targets = targets[:,1:]
                # @hijung - same as training
                _targets = [after_decode(decoder.decode(t)) for t in _targets] 
                _y_hats = [after_decode(decoder.decode(y_)) for y_ in y_hats]
                # _y_hats_lm = [after_decode(decoder_with_lm.decode(y_.detach().cpu().numpy())) for y_ in y_hats]
                # _y_hats_lm = [after_decode(
                #     decoder_with_lm.decode(y_.detach().cpu().numpy()).strip()
                # ) for y_ in outputs.log_softmax(dim=-1)]

                with multiprocessing.get_context('fork').Pool(16) as pool :
                    _y_hats_lm = decoder_with_lm.decode_batch(pool, outputs.log_softmax(dim=-1).detach().cpu().numpy(), beam_width=80)

                _y_hats_lm = [after_decode(i.strip()) for i in _y_hats_lm]


            # _y_hats = [after_decode(
            #                         simple_decoder.decode(y_,
            #                         skip_special_tokens=True,
            #                         )) for y_ in y_hats]
                # _targets = [decoder.decode(t) for t in _targets] 
                # _y_hats = [after_decode(
                #     decoder.decode(y_,
                #     skip_special_tokens=True,
                #     )) for y_ in y_hats]

                cer = cer_metric.compute(references=_targets, predictions=_y_hats)
                cer_ls += [cer]
                # wer = wer_metric(targets[:, 1:], y_hats)
                wer = 0
                print(f'[INFO] VALIDATING step : {cnt:4d}/{len(dataloader):4d}, mean_loss : {epoch_loss_total/cnt:.6f}, mean_cer : {np.mean(cer_ls):.2f}, current_cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m')
                # print(log_format.format(
                #     cnt, len(dataloader), loss,
                #     cer, elapsed, epoch_elapsed, train_elapsed,
                #     # optimizer.get_lr(),
                # ))
                print('-'*10)
                print("[Target]", _targets[0])
                print("[Not LM]",_y_hats[0])
                print("[WithLM]", _y_hats_lm[0])
                print('-'*100)


            cnt += 1
            torch.cuda.empty_cache()

        return model, epoch_loss_total/len(dataloader), np.mean(cer_ls)
    

def decoder_withoud_LM(y_hats, targets, vocab):
    '''
    LM decoder, 혹은 그냥 decoder 구현 필요성이 있음.
    '''
    y_hat_decoded = vocab.label_to_string(y_hats.cpu().detach().numpy())
    targets_decoded = vocab.label_to_string(targets.cpu().detach().numpy())
    return y_hat_decoded, targets_decoded


def decoder_with_LM():
    pass
