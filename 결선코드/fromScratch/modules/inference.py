import os
import numpy as np
import torchaudio

import torch
import torch.nn as nn
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio
from modules.model.deepspeech2 import DeepSpeech2
import librosa
import sklearn
from modules.data import (
    reduce_noise,
    remove_noise_data,
    detect_silence,
    inferDataset,
)
from transformers import Wav2Vec2CTCTokenizer

import json
from pyctcdecode import build_ctcdecoder
import os
import pandas as pd
from modules.data import (
    CustomDataset_2,
    CustomPadFill_2
)
import time
import multiprocessing

import torch

from torch.utils.data import (DataLoader, SequentialSampler, BatchSampler)

import glob


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)


def single_infer(model, audio_path):
    device = 'cuda'
    feature = parse_audio(audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    model.device = device
    y_hats, _ = model.recognize(feature.unsqueeze(0).to(device), input_length)
    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())

    return sentence



def load_simple_decoder(
        json_fn = os.path.join(os.getcwd(), 'unit2id.json')
        ):
    simple_decoder = Wav2Vec2CTCTokenizer(json_fn,
                                          bos_token = '<s>',
                                          eos_id = '</s>',
                                          pad_id = '[pad]',
                                          word_delimiter_token = ' ')
    return simple_decoder


def custom_oneToken_infer(model, features, feature_lengths, targets, vocab):
    model.eval()
    if next(model.parameters()).is_cuda == False:
        model.to('device')

    with torch.no_grad():
        y_hats, _ = model(features, feature_lengths)
        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        answer_sentence = vocab.label_to_string(targets.cpu().detach().numpy())
    return sentence, answer_sentence


def custom_oneToken_infer_validation(model, feature, feature_length, vocab):
    model.eval()
    if next(model.parameters()).is_cuda == False:
        model.to("device")

    with torch.no_grad():
        y_hats, _ = model(feature.unsqueeze(0).to('cuda'), feature_length)
        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    
    return sentence

def custom_oneToken_infer_for_testing(model, audio_path, vocab, config):
    if next(model.parameters()).is_cuda == False:
        model.to('device')
    model.eval()
    device = 'cuda'

    feature = inference_wav2image_tensor(audio_path, config)
    input_length = torch.IntTensor([feature.shape[-1]]).to(device)

    print(feature.unsqueeze(0).transpose(1,2).shape, input_length.shape, input_length)
    outputs, _ = model(feature.unsqueeze(0).transpose(1,2).to(device), input_length)

    y_hats = outputs.max(-1)[1]
    print(y_hats)
    decoder = load_simple_decoder(config.vocab_json_fn)
    sentence = decoder.decode(y_hats[0], skip_special_tokens=True)
    # sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    print('single infer done')
    return sentence

def custom_oneToken_infer_for_testing_dataloader(model, features, feature_lens, config):
    if next(model.parameters()).is_cuda == False:
        model.to('cuda')
    model.eval()
    device = 'cuda'

    output, _ = model(features.transpose(1,2).to(device),feature_lens)
    y_hats = output.max(-1)[1]
    decoder = load_simple_decoder(config.vocab_json_fn)
    sentences = decoder.batch_decode(_y_hats)
    return sentences



def inference_wav2image_tensor(path, config):
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

    mfcc = librosa.feature.mfcc(
        y = audio, 
        sr=config.sample_rate, 
        n_mfcc=config.n_mels, 
        n_fft=400, 
        hop_length=160
    )
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    mfcc = torch.tensor(mfcc, dtype=torch.float)
    return mfcc


# NEW - @hijung - 2023.10.31
def inference(path, model, config, **kwargs):
    def after_decode(str_ls:str):
        return str_ls.replace("<eos>", "")

    #####################
    # Build Model
    #####################
    device = 'cuda'
    if next(model.parameters()).is_cuda == False:
        model.to(device)
    model.eval()
    simple_decoder = Wav2Vec2CTCTokenizer(config.vocab_json_fn,
                                        bos_token = '<sos>',
                                        eos_token = '<eos>',    # labels.csv 불러와서 하는거기에...
                                        pad_token = '<pad>',
                                        word_delimiter_token = ' ')
    #####################
    # Build Dataloader
    #####################
    path_listdir = os.listdir(path)
    path_listdir = [os.path.join(path, i) for i in path_listdir]

    df_test = pd.DataFrame({'filename' : path_listdir, 
                            'text' : ['']*len(path_listdir)})
    test_dataset = CustomDataset_2(
                        df_test,
                        config=config,
                        )
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=config.batch_size*3,
        collate_fn=CustomPadFill_2(0,config),
        num_workers=config.num_workers,
        drop_last=False, # for inference
        shuffle=False # for inference

    )

    #####################
    # Start Inference
    #####################
    total_file_nm = []
    total_y_hat = []
    cnt = 0
    begin_time = time.time()

    with torch.no_grad():
        for inputs, input_lengths, file in test_loader: # input_lengths : audio seq length, target_length : token length
            _start = time.time()

            inputs = inputs.to(device)
            outputs, output_lengths = model(inputs, input_lengths)
            ######### 이부분 accumulate으로 변경

            y_hats = torch.argmax(outputs.log_softmax(-1), dim=-1)

            # @hijung - same as training decoding
            _y_hats = [after_decode(simple_decoder.decode(y_)) for y_ in y_hats]

            total_y_hat += _y_hats
            total_file_nm += file
            cnt += 1
            _end = time.time()

            torch.cuda.empty_cache()

    results = [{'filename': i_file.replace("\\","/").split('/')[-1], 'text':i_y_hat} for i_file, i_y_hat in list(zip(total_file_nm, total_y_hat))]
    return sorted(results, key=lambda x: x['filename'])





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
    

def inference_1(path, model, config, **kwargs):
    def after_decode(str_ls:str):
        return str_ls.replace("<eos>", "")
    
    device = 'cuda'
    if next(model.parameters()).is_cuda == False:
        model.to(device)
    model.eval()

    decoder_with_lm = decoderWithLM(config.vocab_json_fn,
                                    config.lm_fn)

    #####################
    # Build Dataloader
    #####################
    path_listdir = os.listdir(path)
    path_listdir = [os.path.join(path, i) for i in path_listdir]

    df_test = pd.DataFrame({'filename' : path_listdir, 
                            'text' : ['']*len(path_listdir)})
    test_dataset = CustomDataset_2(
                        df_test,
                        config=config,
                        )
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=config.batch_size*3,
        collate_fn=CustomPadFill_2(0,config),
        num_workers=config.num_workers,
        drop_last=False, # for inference
        shuffle=False # for inference
    )


    #####################
    # Start Inference
    #####################
    total_file_nm = []
    total_y_hat = []
    cnt = 0
    begin_time = time.time()

    with torch.no_grad():
        for inputs, input_lengths, file in test_loader: # input_lengths : audio seq length, target_length : token length
            _start = time.time()

            inputs = inputs.to(device)
            outputs, output_lengths = model(inputs, input_lengths)


            with multiprocessing.get_context('fork').Pool(16) as pool :
                _y_hats = decoder_with_lm.decode_batch(pool, outputs.log_softmax(dim=-1).detach().cpu().numpy(), beam_width=80)

            _y_hats = [after_decode(i.strip()) for i in _y_hats]


            total_y_hat += _y_hats
            total_file_nm += file
            cnt += 1
            _end = time.time()
            print(f'{cnt} : {_end-_start:.4f} / for_loop_elasped : {_end-begin_time:.4f}')

            torch.cuda.empty_cache()

    results = [{'filename': i_file.replace("\\","/").split('/')[-1], 'text':i_y_hat} for i_file, i_y_hat in list(zip(total_file_nm, total_y_hat))]
    return sorted(results, key=lambda x: x['filename'])







def customInference(path, model, config, **kwargs):
    model.eval()
    _ = ''

    def make_datafrmae(path, config):
        listdir = glob(os.path.join(path, '*'))
        df = pd.DataFrame(listdir, columns = ['filename'])
        return df
    

    df = make_datafrmae(path, config)
    _inferDataset = inferDataset(df)
    inferenceDataLoader = DataLoader(_inferDataset,
                                     batch_sampler=16,
                                     shuffle=False,
                                     collate_fn=inferCustomPadFill(0,config),
                                     num_workers=config.num_workers,
                                     drop_last=False)

    results = []
    print(next(iter(inferenceDataLoader)))
    for feature, filename, feature_len in inferenceDataLoader:
        sentences = custom_oneToken_infer_for_testing_dataloader(model, feature, feature_len, config)
        for _f, _s in zip(filename, sentences):
            results += [
                {
                    'filename' : _f.split('/')[-1],
                    'text' : _s.replace("<eos>", "")
                }
            ]

    return sorted(results, key=lambda x: x['filename'])


