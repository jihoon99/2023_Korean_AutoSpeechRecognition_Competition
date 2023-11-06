import os
import random
import warnings
import time
import json
import argparse
from glob import glob

from datasets import load_dataset, Audio
from transformers import AutoProcessor
from transformers import (
    AutoProcessor, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    Wav2Vec2CTCTokenizer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from evaluate import load

from jiwer import wer
from transformers import Wav2Vec2CTCTokenizer

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, SequentialSampler, BatchSampler)
from torch.optim.optimizer import Optimizer
from torch import optim

import pandas as pd
import pickle
from datasets import (Dataset, load_metric, DatasetDict)
import librosa
from noisereduce import reduce_noise
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from modules.custom_data import (
    CustomDataset_1,
    CustomDataset_2,
    CustomPadFill,
    CustomSpeechSeq2SeqWithPadding,
    huggingFacePadFill,
    huggingFacePadFill_2
)

import torch.nn.functional as F

from modules.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)

import logging

import nova
from nova import DATASET_PATH


def testing_pred_lengths(preds):
    preds = F.log_softmax(preds, dim=-1)
    preds_lengths = torch.sum(torch.ones_like(preds[:,:,0]).long(), dim=-1)
    return preds_lengths

def training(model, dataloader, tokenizer, processor, criterion, cer_metric, train_begin_time, config, epoch):
    model.train()
    total_num = 0
    epoch_loss_total = 0.
    cer_ls = []

    print(f'[INFO] TRAINING Start')
    epoch_begin_time = time.time()
    cnt = 0
    result = []

    for inputs, targets, input_lengths, target_lengths, audios, file, decoder_attn_mask in dataloader: # input_lengths : audio seq length, target_length : token length
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        # targets = targets.to(device)
        decoder_input_ids = targets[:,:-1].to(device)
        labels = targets[:,1:].to(device)

        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        decoder_attn_mask = decoder_attn_mask[:,1:].to(device)

        if config.pretrained_version == 'wav2vec':
            outputs, output_lengths = model(inputs, input_lengths)
            input_lengths = testing_pred_lengths(outputs)

            loss = criterion(
                outputs.log_softmax(-1).transpose(0, 1),
                targets[:, 1:],
                tuple(input_lengths),
                tuple(target_lengths-1)
                )


            y_hats = torch.argmax(outputs.logits, dim=-1)
            # batch 128 크다 그러니까 : cumulate backward step 방법론 생각해봄직함. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)

            total_num += int(input_lengths.sum())
            epoch_loss_total += loss.item()

            if cnt % config.print_every == 0:

                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0    # 아 초, 단위로 한거구나.
                train_elapsed = (current_time - train_begin_time) / 3600.0  # 시간 단위로 변환한거구나.
                _targets = targets[:,1:]
                _targets = [tokenizer.decode(t) for t in _targets] 
                _y_hats = [tokenizer.decode(y_) for y_ in y_hats]
                cer = cer_metric.compute(references=_targets, predictions=_y_hats)
                cer_ls += cer
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

        elif config.pretrained_version == 'whisper':
            # options = dict(language='Korean', beam_size=5, best_of=5)
            # transcribe_options = dict(task="transcribe", **options)
            optimizer.zero_grad()

            outputs = model(
                input_features = inputs,
                decoder_input_ids = decoder_input_ids,
                labels = labels,
                decoder_attention_mask = decoder_attn_mask
            )

            logit = outputs.logits
            loss = outputs.loss

            forced_decoder_ids = processor.get_decoder_prompt_ids(language="Korean", task="transcribe")
            # outputs = F.log_softmax(outputs, dim=-1)

            # loss = criterion(
            #     outputs.transpose(1,2).contiguous().view(-1, outputs.shape[-1]),
            #     targets.contiguous().view(-1)
            # )
            epoch_loss_total += loss.item()

            loss.backward()
            optimizer.step()



            y_hat = torch.argmax(logit, dim=-1)

            # gen_y_hat = model.generate(inputs=inputs, forced_decoder_ids=forced_decoder_ids)     # [bs, seq]

            ################################## for local only
            # result += [
            #     {
            #         'answer' : _t,
            #         'student' : _s,
            #         'generate' : _g,
            #         # 'cer' : cer_metric.compute(references=[_t], predictions=[_g])
            #     } for _t, _s, _g in zip(tokenizer.batch_decode(targets,  skip_special_tokens=True),
            #                             tokenizer.batch_decode(y_hat, skip_special_tokens=True),
            #                             processor.batch_decode(gen_y_hat, skip_special_tokens=True))
            # ]


            if cnt % config.print_every == 1:

                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0    # 아 초, 단위로 한거구나.
                train_elapsed = (current_time - train_begin_time) / 3600.0  # 시간 단위로 변환한거구나.
                # _targets = targets[:,1:]
                _targets = [tokenizer.batch_decode(targets, skip_special_tokens=True)[0]]
                y_hat = torch.argmax(logit, dim=-1)
                _y_hats = [tokenizer.batch_decode(y_hat, skip_special_tokens=True)[0]]
                # _y_hats = [tokenizer.decode(y_) for y_ in outputs]
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="Korean", task="transcribe")
                y_hat = model.generate(inputs=inputs, forced_decoder_ids=forced_decoder_ids)     # [bs, seq]
                y_hat = [processor.batch_decode(y_hat, skip_special_tokens=True)[0]]

                # y_hat1 = model.generate(inputs=inputs)     # [bs, seq]
                # y_hat1 = [processor.batch_decode(y_hat1, skip_special_tokens=True)[0]]

                cer = cer_metric.compute(references=_targets, predictions=y_hat)
                cer_ls += [cer]
                #wer = wer_metric(targets[:, 1:], y_hats)
                wer = 0

                print(f'[INFO] TRAINING step : {cnt:4d}/{len(dataloader):4d}, mean_loss : {epoch_loss_total/cnt:.6f}, mean_cer : {np.mean(cer_ls):.2f}, current_cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m {train_elapsed:.2f}h')
                print('-'*100)
                print('answer : ',_targets[0])
                print('student : ',_y_hats[0])
                print('generation : ',y_hat[0])
                print('-'*100)

                # logging.info(f'TRAINING epoch-{epoch}step : {cnt:4d}/{len(dataloader):4d}, loss : {loss:.6f}, cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m {train_elapsed:.2f}h')
                # logging.info(f"TRAIN epoch-{epoch} {cnt} loss : {loss}")
                # logging.info(f"TRAIN epoch-{epoch} {cnt} answer : {_targets[0]}")
                # logging.info(f"TRAIN epoch-{epoch} {cnt} student : {_y_hats[0]}")
                # logging.info(f"TRAIN epoch-{epoch} {cnt} generate : {y_hat[0]}")
                
                # print(log_format.format(
                #     cnt, len(dataloader), loss,
                #     cer, elapsed, epoch_elapsed, train_elapsed,
                #     # optimizer.get_lr(),
                # ))
            cnt += 1
            torch.cuda.empty_cache()
    # result = pd.DataFrame(result)
    # result.to_pickle(f"/data/asr/pre_trained/log/train_{epoch}.pkl")
    return model, epoch_loss_total/len(dataloader), np.mean(cer)


def validating(model, dataloader, tokenizer, processor, criterion, cer_metric, train_begin_time, config, epoch):
    model.eval()

    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] VALIDATING Start')
    epoch_begin_time = time.time()
    cnt = 0
    cer_ls = []
    result = []

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths, audios, file, decoder_attn_mask in dataloader: # input_lengths : audio seq length, target_length : token length
            # input_lengths : audio seq length, target_length : token length
            begin_time = time.time()

            inputs = inputs.to(device)
            # targets = targets.to(device)
            decoder_input_ids = targets[:,:-1].to(device)
            labels = targets[:,1:].to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            decoder_attn_mask = decoder_attn_mask[:,1:].to(device)

            if config.pretrained_version == 'wav2vec':
                pass
            elif config.pretrained_version == 'whisper':
                # options = dict(language='Korean', beam_size=5, best_of=5)
                # transcribe_options = dict(task="transcribe", **options)

                outputs = model(
                    input_features = inputs,
                    decoder_input_ids = decoder_input_ids,
                    labels = labels,
                    decoder_attention_mask = decoder_attn_mask
                )

                logit = outputs.logits
                loss = outputs.loss
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="Korean", task="transcribe")




            y_hat = torch.argmax(logit, dim=-1)

            # gen_y_hat = model.generate(inputs=inputs, forced_decoder_ids=forced_decoder_ids)     # [bs, seq]

            # result += [
            #     {
            #         'answer' : _t,
            #         'student' : _s,
            #         'generate' : _g,
            #         # 'cer' : cer_metric.compute(references=[_t], predictions=[_g])
            #     } for _t, _s, _g in zip(tokenizer.batch_decode(targets,  skip_special_tokens=True),
            #                             tokenizer.batch_decode(y_hat, skip_special_tokens=True),
            #                             processor.batch_decode(gen_y_hat, skip_special_tokens=True))
            # ]





            if cnt % config.print_every == 1:
                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0


                _targets = [tokenizer.batch_decode(targets, skip_special_tokens=True)[0]]
                
                y_hat = torch.argmax(logit, dim=-1)
                _y_hats = [tokenizer.batch_decode(y_hat, skip_special_tokens=True)[0]]

                forced_decoder_ids = processor.get_decoder_prompt_ids(language="Korean", task="transcribe")
                y_hat = model.generate(
                    inputs=inputs, 
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=5,
                    num_return_sequences=1)     # [bs, seq]
                y_hat = [processor.batch_decode(y_hat, skip_special_tokens=True)[0]]


                cer = cer_metric.compute(references=_targets, predictions=y_hat)
                cer_ls += [cer]
                #wer = wer_metric(targets[:, 1:], y_hats)
                wer = 0

                print(f'[INFO] VALIDATING step : {cnt:4d}/{len(dataloader):4d}, mean_loss : {epoch_loss_total/cnt:.6f}, mean_cer : {np.mean(cer_ls):.2f}, current_cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m')
                print('-'*100)
                print('answer : ',_targets[0])
                print('student : ',_y_hats[0])
                print('generation : ',y_hat[0])
                print('-'*100)

                # logging.info('VALIDATING epoch-{epoch} step : {cnt:4d}/{len(dataloader):4d},  cer : {cer:.2f}, wer : {wer:.2f}, elapsed : {elapsed:.2f}s {epoch_elapsed:.2f}m')
                # logging.info(f"VALID epoch-{epoch} {cnt} loss : {loss}")
                # logging.info(f"VALID epoch-{epoch} {cnt} answer : {_targets[0]}")
                # logging.info(f"VALID epoch-{epoch} {cnt} student : {_y_hats[0]}")
                # logging.info(f"VALID epoch-{epoch} {cnt} generate : {y_hat[0]}")

            cnt += 1
            torch.cuda.empty_cache()


        # result = pd.DataFrame(result)
        # result.to_pickle(f"/data/asr/pre_trained/log/valid_{epoch}.pkl")

        return model, epoch_loss_total/len(dataloader), np.mean(cer_ls)

@dataclass
class DataCollatorCTCWithPadding:


    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def remove_noise_data(np_wav, ratio=16_000):
    return reduce_noise(y = np_wav, sr=ratio, stationary=False)

def detect_silence(pcm, audio_threshold = 0.0075, min_silence_len = 3, ratio=16000, make_silence_len=1):
    if len(pcm) < min_silence_len*ratio:
        return pcm
    
    b = np.where((abs(pcm) > audio_threshold) == True)[0] # 소리가 나는 부분
    c = np.concatenate(([0], b[:-1]), axis=0)

    starts = c[(b-c)>min_silence_len*ratio]               # 소리가 안나는 부분 시작
    ends = b[(b-c)>min_silence_len*ratio]

    if len(ends) == 0:
        return pcm
    else:
        non_masking = np.array([True]*len(pcm))
        for (s,e) in zip(starts, ends):
            non_masking[s:e+1] = False
            non_masking[e-make_silence_len*ratio:e+1] = True
        
        return pcm[non_masking]

def wav2image_tensor(path, config):
    audio, sr = librosa.load(path, sr=config.sample_rate)
    audio, _ = librosa.effects.trim(audio)

    if config.remove_noise:
        audio = remove_noise_data(audio)

    if config.del_silence:
        audio = detect_silence(
            audio,
            audio_threshold=config.audio_threshold,
            min_silence_len=config.min_silence_len,
            ratio = config.sample_rate,
            make_silence_len=config.make_silence_len
            )
    return audio

def prepare_dataset(batch, processor, tokenizer, config, file_fn_col='filename', target_col = 'text'):
    if config.preprocess_wav:
        audio, sr = librosa.load(batch[file_fn_col], sr=config.sample_rate)
    if config.remove_noise:
        audio = remove_noise_data(audio)
        audio, _ = librosa.effects.trim(audio)

    if config.del_silence:
        audio = detect_silence(
            audio,
            audio_threshold=config.audio_threshold,
            min_silence_len=config.min_silence_len,
            ratio = config.sample_rate,
            make_silence_len=config.make_silence_len
            )
    
    # input audio array로부터 log-Mel spectrogram 변환
    # batched output is "un-batched" to ensure mapping is correct
    input_features = processor(audio, sampling_rate=config.sample_rate, return_tensors = 'pt', padding='longest').input_features[0]
    input_len = len(input_features)

    if config.pretrained_version == 'wav2vec':
        with processor.as_target_processor():
            label = processor(batch[target_col]).input_ids
    elif config.pretrained_version == 'whisper':
        # _label = processor(batch[target_col]).input_ids
        label = tokenizer(batch[target_col]).input_ids

    return {'filename': batch[file_fn_col], 'text':batch[target_col], 'input_features':input_features, 'feature_len':input_len, 'input_ids':label}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



def read_dataframe(csv_fn, poc = False):
    label_df = pd.read_csv(csv_fn)
    if poc : 
        return label_df.iloc[:1000]
    return label_df

def add_absolute_path(csv:pd.DataFrame, root_path):
    csv['filename'] = root_path + '/'+ csv['filename']
    return csv

def train_valid_split(df, valid_ratio=0.15):
    train=df.sample(
        frac=1-valid_ratio,
        random_state=200)
    valid=df.drop(train.index)
    return train, valid

def from_pandas_to_datasets(
        csv : pd.DataFrame):
    return Dataset.from_pandas(csv)




def from_pandas_to_datasets(
        csv : pd.DataFrame):
    return Dataset.from_pandas(csv)


def get_optimizer(model: nn.Module, config):
    '''    supported_optimizer = {
        'adam': optim.Adam,
        # 'Radam' : optim.RAdam,  # custom.optim.RAdam 으로 바꿔야함.
    }'''

    if config.optimizer == 'adam':
        _optim = optim.Adam
    elif config.optimizer == 'Radam':
        pass ################################################################## 구현 필요
    elif config.optimizer == 'RMSprop':
        _optim = optim.RMSprop
    
    if config.weight_decay:
        return _optim(
            model.parameters(),
            lr=config.init_lr,
            weight_decay=config.weight_decay,
        )
    else:
        return _optim(
            model.parameters(),
            lr=config.init_lr,
        )

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))



def bind_model(model, config, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))

        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

        print('Model and config saved')


    def load(path, *args, **kwargs):
        with open(os.path.join(path, "config.pkl"), 'rb') as f:
            config = pickle.load(f)

        model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_model_name) # 추가 JH
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        model.to(device)
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # NEW - @hijung - 2023.10.30    
    def infer(path, **kwargs):
        return inference(path, model, config)
    nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.


def inference(path, model, config, **kwargs):
    ##################
    # build tokenizer, processor
    #################
    tokenizer = WhisperTokenizer.from_pretrained(config.pretrained_model_name, language="Korean", task="transcribe") # finetuning ... transcribe for speech recognition  // https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/whisper/tokenization_whisper.py#L215
    processor = WhisperProcessor.from_pretrained(config.pretrained_model_name, language="Korean", task="transcribe") # tokenizer, feature_extractor
    model.eval()

    path_listdir = os.listdir(path)
    path_listdir = [os.path.join(path, i) for i in path_listdir]

    df_test = pd.DataFrame({'filename' : path_listdir, 
                            'text' : ['']*len(path_listdir)})

    test_dataset = CustomDataset_2(
            df = df_test,
            processor = processor, 
            tokenizer = tokenizer,
            config = config,
        )
    
    test_dataLoader = DataLoader(
        test_dataset,
        batch_size = config.batch_size*3,
        num_workers = config.num_workers,
        collate_fn = huggingFacePadFill_2(processor, config),            
        shuffle = False,    # for inference 
        drop_last = False   # for inference 
        )
    
    #################
    # start inference
    #################
    total_file_nm = []
    total_y_hat = []
    begin_time = time.time()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="Korean", task="transcribe")

    with torch.no_grad():
        for inputs, input_lengths, audios, file in test_dataLoader: # input_lengths : audio seq length, target_length : token length
            inputs = inputs.to(device)
            y_hat = model.generate(
                inputs=inputs, 
                forced_decoder_ids=forced_decoder_ids,
                num_beams=5,
                num_return_sequences=1)     # [bs, seq]

            y_hat = [i.strip() for i in processor.batch_decode(y_hat, skip_special_tokens=True)]
            # y_hat = y_hat.detach().cpu().reshape(-1).tolist()
            
            total_y_hat += y_hat
            total_file_nm += file

        current_time = time.time()
        elapsed = current_time - begin_time

    results = [{'filename': i_file.replace("\\","/").split('/')[-1], 'text':i_y_hat} for i_file, i_y_hat in list(zip(total_file_nm, total_y_hat))]
    return sorted(results, key=lambda x: x['filename'])



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')

    # logging.basicConfig(filename='whisper_logging_2',level=logging.INFO)
    

    args = argparse.ArgumentParser()
    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')   # mode - related to nova system. barely change
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)
    # Parameters 
    args.add_argument('--version', type=str, default='train')                     # train, valid, POC
    args.add_argument("--pretrained_version", type=str, default='whisper')      # whisper, wav2vec 
    args.add_argument('--pretrained_model_name', default="openai/whisper-base") # "openai/whisper-base"
    args.add_argument("--use_dataset", default=False)
    args.add_argument("--if_use_dataloader", default=True)
    args.add_argument("--preprocess_wav", default=True)         # df의 filename이 wav일 경우.

    args.add_argument('--loss_fn', default='cross_entropy')            # nll, ctcloss, cross_entropy

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=30)
    args.add_argument('--batch_size', type=int, default=16)

    args.add_argument('--save_result_every', type=int, default=2) # 2 
    args.add_argument('--checkpoint_every', type=int, default=1)  # save model every 
    args.add_argument('--print_every', type=int, default=50)
    
    # Data Processing
    args.add_argument('--ignore_n_character', default=True)
    args.add_argument('--n_character', default=4)


    args.add_argument('--audio_extension', type=str, default='wav')
    
    args.add_argument("--del_silence", type=bool, default=True)
    args.add_argument("--remove_noise", type=bool, default=True)
    args.add_argument("--audio_threshold", type=float, default=0.0075) # 에선 대회 3에서는 0.0885
    args.add_argument("--min_silence_len", type=float, default=3)
    args.add_argument("--make_silence_len", type=float, default=1)
    
    #MFCC hardCoded
    args.add_argument("--mfcc_max_len", default=1600)         # conformer였을때는 1600, 이였음. 근데 whisper는 모델의 특성상 3000을 받아야함.

    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    # DataLoader    
    args.add_argument('--num_workers', type=int, default=16)  # 16
    args.add_argument('--num_threads', type=int, default=16)  # 16
    
    # optimizer
    args.add_argument('--optimizer', type=str, default='RMSprop')       # adam, rmsprop, Radam

    # Optimizer lr scheduler options
    args.add_argument("--constant_lr", default=5e-5) #default=5e-5)  # when this is False:   아래의 옵션들이 실행됨.
    args.add_argument('--use_lr_scheduler', type=bool, default=False) ## 
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--warmup_steps', type=int, default=1000)
    # args.add_argument('--weight_decay', default=1e-05)
    args.add_argument('--weight_decay', default=False)

    # explode 예방
    args.add_argument('--max_grad_norm', type=int, default=400) ######################## 수정이 필요함.

    # check
    args.add_argument('--reduction', type=str, default='mean')        # check
    args.add_argument('--total_steps', type=int, default=200000)


    # args.add_argument('--architecture', type=str, default='conformer') # check
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')           # check
    args.add_argument('--max_len', type=int, default=400)              #############################이부분도 ???
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--use_bidirectional', type=bool, default=True) # maybe decoder aggregator

    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)  # check
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)   # check
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)  # check
    args.add_argument('--joint_ctc_attention', type=bool, default=False)   # check
    config = args.parse_args()


    if config.version == 'POC':
        config.num_epochs = 5

    warnings.filterwarnings('ignore')
    # seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # thread
    # if hasattr(config, "num_threads") and int(config.num_threads) > 0:
    #     torch.set_num_threads(config.num_threads)
    # load pre_trained_model


    print("setting wer metric!!!")
    wer_metric = load_metric("wer")


    print("Load Pre-trained Model")
    print(f"loading {config.pretrained_version}")

    if config.pretrained_version == 'whisper':
        '''tokenizer : special token들을 부여한다. 
        
        input_str = "저는 서울중앙지검 지능범죄수사팀 최인호 검사입니다."
        labels = tokenizer(input_str).input_ids
        decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
        decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

        print(f"Input:                 {input_str}")
        print(f"Decoded w/ special:    {decoded_with_special}")
        print(f"Decoded w/out special: {decoded_str}")
        print(f"Are equal:             {input_str == decoded_str}")
        

        Input:                 저는 서울중앙지검 지능범죄수사팀 최인호 검사입니다.
        Decoded w/ special:    <|startoftranscript|><|ko|><|transcribe|><|notimestamps|>저는 서울중앙지검 지능범죄수사팀 최인호 검사입니다.<|endoftext|>
        Decoded w/out special: 저는 서울중앙지검 지능범죄수사팀 최인호 검사입니다.
        Are equal:             True
        '''
        feature_extractor =  WhisperFeatureExtractor.from_pretrained(config.pretrained_model_name)  # log-Mel로 변환해주는 것. from audio
        tokenizer = WhisperTokenizer.from_pretrained(config.pretrained_model_name, language="Korean", task="transcribe") # finetuning ... transcribe for speech recognition  // https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/whisper/tokenization_whisper.py#L215
        processor = WhisperProcessor.from_pretrained(config.pretrained_model_name, language="Korean", task="transcribe") # tokenizer, feature_extractor
        # feature extractor와 tokenizer를 한번에 묶은것,
        model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_model_name).to(device)
        # 이부분 고쳐야함.
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor) ###############################


    elif config.pretrained_version == 'wav2vec':
        processor = Wav2Vec2Processor.from_pretrained(config.pretrained_model_name)
        model = Wav2Vec2ForCTC.from_pretrained(config.pretrained_model_name,
            pad_token_id=processor.tokenizer.pad_token_id).to(device)

        label_path = 'labels.csv'
        label_df = pd.read_csv(label_path).drop('freq', axis=1)
        char2idx = label_df.set_index("char").to_dict()['id']
        idx2char = label_df.set_index("id").to_dict()['char']
        import json

        with open("voacb.json", 'w') as vocab_file:
            json.dump(char2idx, vocab_file)

        tokenizer = Wav2Vec2CTCTokenizer("vocab.json", pad_token='<pad>', word_delimiter_token=' ') 
        

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    

    print(" ############################################## number of params ##############################################")
    print(count_parameters(model))
    print(" ##############################################################################################################")


    optimizer = get_optimizer(model, config)
    bind_model(model, optimizer=optimizer, config=config)
    print("Load model success!!")


    if config.pause:
        nova.paused(scope=locals())


    if config.mode == 'train':
        root_dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        df = read_dataframe(os.path.join(DATASET_PATH, 'train', 'train_label'))
        df = add_absolute_path(df, root_dataset_path)
        
        ##### JH Local 임.. 바궈야함.
        # df = pd.read_pickle("/data/asr/audio_test.pkl")   # filename, text
        # df['filename'] = '/data/asr/test_data/' + df['filename'].astype(str) + '.wav'
        print(df.head())  # filename, text

        if config.version == 'POC':
            df = df.iloc[:1000]


        train, valid = train_valid_split(df)
        print(train.head())

        ds_train = Dataset.from_pandas(train)
        ds_test = Dataset.from_pandas(valid)

        ds_dict = {'train' : ds_train,
                'test' : ds_test}

        df_dataset = DatasetDict(ds_dict)

        if config.pretrained_version == 'wav2vec':
            model.freeze_feature_encoder() #

        ds_train = df_dataset['train']
        ds_test = df_dataset['test']

        if config.if_use_dataloader:

            ds_train = ds_train.to_pandas()
            ds_test = ds_test.to_pandas()

            train_dataset = CustomDataset_1(
                df = ds_train,
                processor = processor, 
                tokenizer = tokenizer,
                config = config,
            )

            valid_dataset = CustomDataset_1(
                df = ds_test,
                processor = processor, 
                tokenizer = tokenizer,
                config = config,
            )


            if config.pretrained_version == 'whisper':
                collate_fn = CustomSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)
            elif config.pretrained_version == 'wav2vec':
                collate_fn = 'pass' ######################################
            collate_fn = CustomPadFill(50257,config)


            train_dataLoader = DataLoader(
                train_dataset,
                batch_size = config.batch_size,
                shuffle=True,
                # collate_fn=collate_fn,
                collate_fn=huggingFacePadFill(processor, config),
                num_workers=config.num_workers,
                drop_last=True
                )
            
            _a = next(iter(train_dataLoader))
            label_features = [{"input_ids": feature[1]} for feature in _a]

            print(_a)
            
            valid_dataLoader = DataLoader(
                valid_dataset,
                batch_size = config.batch_size,
                shuffle=True,
                # collate_fn=collate_fn,
                collate_fn=huggingFacePadFill(processor, config),
                num_workers=config.num_workers,
                drop_last=True
                )
            

            print("#"*100)
            _seqs, _targets, _seq_lengths, _target_lengths, _audios, _file, _decoder_attn_mask  = next(iter(train_dataLoader))
            print(f"TRAINING data shape")
            print(f'TRAINING DATA : seq : {_seqs.shape}')
            print(f'TRAINING DATA : targets : {_targets.shape}')
            print(f'TRAINING DATA : seq lengths : {_seq_lengths}')
            print("#"*100)
            print(f"number of dataset : {len(train_dataLoader)}")
            print(_seqs)
            print(_targets)
            print(_seq_lengths)
            print(_target_lengths)
        else:
            df_dataset = df_dataset.map(lambda x: prepare_dataset(x, processor, tokenizer, config), batch_size=1000) #, batched=True) #, num_proc=6) #
            print(df_dataset)
            print(df_dataset['train'].to_pandas().head(5))
            _a = df_dataset['train'].to_pandas().head(5)['input_ids'].iloc[0]
            _b = df_dataset['train'].to_pandas().head(5)['input_ids'].iloc[1]
            print(_a)





        # lr 스케쥴 적용한 것과 아닌것.
        if config.use_lr_scheduler:
            lr_scheduler = get_lr_scheduler(config, optimizer, len(train_dataset)) # learning scheduler 적용했네.
            optimizer = Optimizer(optimizer, lr_scheduler, int(len(train_dataset)*config.num_epochs), config.max_grad_norm)

        if config.pretrained_version == 'whisper':
            pad_id = tokenizer.encoder[tokenizer.eos_token]
        elif config.pretrained_version == 'ctcloss':
            pad_id = tokenizer.pad_id  ##??

        criterion = get_criterion(
            config,
            len(tokenizer),
            pad_id
        ).to(device) # CTC loss


        cer_metric = load("cer")

        train_begin_time = time.time()


        ###################### let's check performance ################################

        print("#### zero shot ###########################")
        if False:
            for feature, target, feature_len, target_len, audil, file in train_dataLoader:
                pass





        for epoch in range(config.num_epochs):
            print('[INFO] Epoch %d start' % epoch)
            model, train_loss, train_cer = training(
                model = model,
                dataloader=train_dataLoader,
                tokenizer=tokenizer,
                processor=processor,
                criterion=criterion,
                cer_metric=cer_metric,
                train_begin_time=train_begin_time,
                config=config,
                epoch = epoch
            )


            # train
            # valid
            model,  valid_loss, valid_cer = validating(
                model = model,
                dataloader=valid_dataLoader,
                tokenizer=tokenizer,
                processor=processor,
                criterion=criterion,
                cer_metric=cer_metric,
                train_begin_time=train_begin_time,
                config=config,
                epoch = epoch
            )

            nova.report(
                summary=True,
                epoch=epoch,
                train_loss=train_loss,
                train_cer=train_cer,
                val_loss=valid_loss,
                val_cer=valid_cer
            )


            model_states = model.state_dict()
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            # torch.save(state, f'/data/asr/pre_trained/saved_model/model_epoch_{epoch}.pt')


            if epoch % config.checkpoint_every == 0:
                nova.save(epoch)

            torch.cuda.empty_cache()
            print(f'[INFO] epoch {epoch} is done')
            if config.version == 'POC':
                # out = custom_oneToken_infer_for_testing(model, df.iloc[0]['filename'], _, config)
                # print(out)
                import time
                # time.sleep(1)
                time.sleep(600)

        print('[INFO] train process is done')
