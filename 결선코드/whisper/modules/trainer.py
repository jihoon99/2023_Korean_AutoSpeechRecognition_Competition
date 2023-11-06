import torch
import numpy as np
import math
from dataclasses import dataclass
import time
from nova import DATASET_PATH

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import torch.nn.functional as F


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

    # log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
    #                           "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] TRAINING Start')
    epoch_begin_time = time.time()
    cnt = 0
    for inputs, targets, input_lengths, target_lengths in dataloader: # input_lengths : audio seq length, target_length : token length
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        # model = model.to(device) # 모델을 불러 올 때 이미 gpu에 올림.
        # print(f'input_lengths : {input_lengths.tolist()}')


        outputs, output_lengths = model(inputs, input_lengths)
        # # print(f'output_lengths : {output_lengths.tolist()}')

        # with autocast():
        #     for _output, _output_length, _target, _target_length in zip(outputs, output_lengths, targets, target_lengths):
        #         _output = _output.unsqueeze(0)
                
        #         without_pad_target = _target[1:_target_length]
        #         without_pad_target = without_pad_target.unsqueeze(0)

        #         loss = criterion(
        #             outputs.transpose(0, 1),
        #             targets[:, 1:],
        #             tuple(_output_length),
        #             tuple(_target_length-1)
        #         )

        #         GradScaler.scale(loss).backward()

        #     GradScaler.step(optimizer)
        #     GradScaler.update()
        #     optimizer.zero_grad()

        def testing_pred_lengths(preds):
            preds = F.log_softmax(preds, dim=-1)
            preds_lengths = torch.sum(torch.ones_like(preds[:,:,0]).long(), dim=-1)
            return preds_lengths

        input_lengths = testing_pred_lengths(outputs)

        loss = criterion(
            outputs.log_softmax(-1).transpose(0, 1),
            targets[:, 1:],
            tuple(input_lengths),
            tuple(target_lengths-1)
            )


        y_hats = torch.argmax(outputs.log_softmax(-1), dim=-1)
        # batch 128 크다 그러니까 : cumulate backward step 방법론 생각해봄직함. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(model)

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()


        # if cnt % config.print_every == 0:
        #     print(f'y_hat : {y_hats[0]}')
        #     print(f'targets : {targets[0]}')


        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0    # 아 초, 단위로 한거구나.
            train_elapsed = (current_time - train_begin_time) / 3600.0  # 시간 단위로 변환한거구나.
            _targets = targets[:,1:]
            _targets = [decoder.decode(t) for t in _targets] 
            _y_hats = [decoder.decode(y_) for y_ in y_hats]
            cer = cer_metric.compute(references=_targets, predictions=_y_hats)
            #wer = wer_metric(targets[:, 1:], y_hats)
            wer = 0

            print(f'[INFO] TRAINING step : {cnt:4d}/{len(dataloader):4d}, loss : {loss:.6f}, cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m {train_elapsed:.2f}h')
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

    return model, epoch_loss_total/len(dataloader), cer


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

    model.eval()

    # log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
    #                           "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] VALIDATING Start')
    epoch_begin_time = time.time()
    cnt = 0

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in dataloader: # input_lengths : audio seq length, target_length : token length
            begin_time = time.time()

            # optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            # model = model.to(device) # 모델을 불러 올 때 이미 gpu에 올림.

            outputs, output_lengths = model(inputs, input_lengths)

            ######### 이부분 accumulate으로 변경


            def testing_pred_lengths(preds):
                preds = F.log_softmax(preds, dim=-1)
                preds_lengths = torch.sum(torch.ones_like(preds[:,:,0]).long(), dim=-1)
                return preds_lengths

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


            if cnt % config.print_every == 0:
                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0
                train_elapsed = (current_time - train_begin_time) / 3600.0

                _targets = targets[:,1:]
                _targets = [decoder.decode(t) for t in _targets] 
                _y_hats = [decoder.decode(y_,
                                          skip_special_tokens=True,
                                          ) for y_ in y_hats]
                cer = cer_metric.compute(references=_targets, predictions=_y_hats)

                # wer = wer_metric(targets[:, 1:], y_hats)
                wer = 0
                
                print(f'[INFO] VALIDATING step : {cnt:4d}/{len(dataloader):4d}, loss : {loss:.6f}, cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m {train_elapsed:.2f}h')
                # print(log_format.format(
                #     cnt, len(dataloader), loss,
                #     cer, elapsed, epoch_elapsed, train_elapsed,
                #     # optimizer.get_lr(),
                # ))
                print('-'*100)
                print(_targets[0])
                print(_y_hats[0])
                print('-'*100)


            cnt += 1
            torch.cuda.empty_cache()

        return model, epoch_loss_total/len(dataloader), cer
    

def decoder_withoud_LM(y_hats, targets, vocab):
    '''
    LM decoder, 혹은 그냥 decoder 구현 필요성이 있음.
    '''
    y_hat_decoded = vocab.label_to_string(y_hats.cpu().detach().numpy())
    targets_decoded = vocab.label_to_string(targets.cpu().detach().numpy())
    return y_hat_decoded, targets_decoded


def decoder_with_LM():
    pass
