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
)
from transformers import Wav2Vec2CTCTokenizer



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
