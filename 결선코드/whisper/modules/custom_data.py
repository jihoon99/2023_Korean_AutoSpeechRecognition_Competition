import sys, os
import threading
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import random
import math
import datasets

from torch import Tensor
from torch.utils.data import (Dataset, Sampler, DataLoader)

import librosa
import sklearn

from noisereduce import reduce_noise
import pandas as pd


class huggingFacePadFill():
    def __init__(self, processor, config):
        self.config = config
        self.processor = processor

    def custom_feature_padding(self, feature, max_len, current_len):
        template = nn.ZeroPad2d((0,max_len-current_len,0,0))
        return template(feature)

    
    # def shuffle_1(self, features, targets, feature_len, target_len, audios, file_path, decoder_attn_mask):
    #     idx = list(range(features.shape[0]))
    #     random.shuffle(idx)

    #     new_features = []
    #     new_targets = []
    #     new_target_len = []
    #     new_feature_len = []
    #     new_audios = []
    #     new_files = []
    #     new_decoder_attn_mask = []

    #     for _idx in idx:
    #         new_features += [features[_idx].unsqueeze(0)]
    #         new_targets += [targets[_idx].unsqueeze(0)]
    #         new_target_len += [target_len[_idx]]
    #         new_feature_len += [feature_len[_idx]]
    #         new_audios += [audios[_idx]]
    #         new_files += [file_path[_idx]]
    #         new_decoder_attn_mask += [decoder_attn_mask[_idx].unsqueeze(0)]

    #     features = torch.cat(new_features, dim=0)
    #     targets = torch.cat(new_targets, dim=0).long()
    #     target_len = torch.IntTensor(new_target_len)
    #     feature_len = torch.IntTensor(new_feature_len)
    #     new_decoder_attn_mask =torch.cat(new_decoder_attn_mask, dim=0).long()

    #     return features, targets, feature_len, target_len, new_audios, new_files, new_decoder_attn_mask
    
    def __call__(self,bs):
        
        targets = []
        features = []
        target_len = []
        feature_len = []
        audios = []
        file_fns = []
        decoder_attn_mask = []

        for feature, target, audio, file in bs:
            # target = torch.tensor(target).long()
            # feature_len += [min(feature.shape[-1], self.config.mfcc_max_len)]
            feature_len += [feature.shape[-1]]
            target_len += [target.shape[-1]] # baseline에서는 -1을 햇음 왜그랬을까?
            targets += [{'input_ids' : target}]
            audios += [audio]
            file_fns += [file]


        if self.config.pretrained_version == 'whisper':
            max_feature_len = 3000
        else:
            max_feature_len = min(max(feature_len), self.config.mfcc_max_len)
        max_target_len = max(target_len)

        for (feature, target, _, _), current_feature_len, current_target_len in zip(bs, feature_len, target_len):
            #targets += [self.custom_target_padding1(target, max_target_len, current_target_len)]
            features += [self.custom_feature_padding(feature, max_feature_len, current_feature_len).unsqueeze(0)]
        features = torch.cat(features, dim=0)


        labels_batch = self.processor.tokenizer.pad(targets, return_tensors="pt")
        targets = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.processor.tokenizer.pad_token_id)


        ### 제발 되라...
        final_feature_len = []
        for _f in feature_len:
            if _f > self.config.mfcc_max_len:
                final_feature_len += [self.config.mfcc_max_len]
            else:
                final_feature_len += [_f]
        feature_len = torch.IntTensor(final_feature_len)
        target_len = torch.IntTensor(target_len)
        decoder_mask = labels_batch.attention_mask


        # features, targets, feature_len, target_len, audios, file_fns, decoder_attn_mask = self.shuffle_1(features, targets, feature_len, target_len, audios, file_fns, decoder_mask)
        # return features, targets, feature_len, target_len, audio, file_fns, decoder_attn_mask
        return features, targets, feature_len, target_len, audios, file_fns, decoder_mask




## @hijung(2023.10.30) - for inference 
class huggingFacePadFill_2():
    def __init__(self, processor, config):
        self.config = config
        self.processor = processor

    def custom_feature_padding(self, feature, max_len, current_len):
        template = nn.ZeroPad2d((0,max_len-current_len,0,0))
        return template(feature)
    
    def __call__(self,bs):
        features = []
        feature_len = []
        audios = []
        file_fns = []

        ### hijunghijung
        for feature, audio, file in bs:
            # target = torch.tensor(target).long()
            # feature_len += [min(feature.shape[-1], self.config.mfcc_max_len)]
            feature_len += [feature.shape[-1]]
            audios += [audio]
            file_fns += [file]

        if self.config.pretrained_version == 'whisper':
            max_feature_len = 3000
        else:
            max_feature_len = min(max(feature_len), self.config.mfcc_max_len)

        for (feature, _, _), current_feature_len in zip(bs, feature_len):
            #targets += [self.custom_target_padding1(target, max_target_len, current_target_len)]
            features += [self.custom_feature_padding(feature, max_feature_len, current_feature_len).unsqueeze(0)]
        features = torch.cat(features, dim=0)

        ### 제발 되라...
        final_feature_len = []
        for _f in feature_len:
            if _f > self.config.mfcc_max_len:
                final_feature_len += [self.config.mfcc_max_len]
            else:
                final_feature_len += [_f]
        feature_len = torch.IntTensor(final_feature_len)
        return features, feature_len, audio, file_fns


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


class CustomPadFill():
    def __init__(self, padding_token, config):
        self.config = config
        self.padding_token = padding_token

    #def custom_target_padding(self, target, max_len, current_len):
    #    template = nn.ZeroPad1d((0,max_len-current_len,0,0))
    #    return template(target)

    def custom_feature_padding(self, feature, max_len, current_len):
        template = nn.ZeroPad2d((0,max_len-current_len,0,0))
        return template(feature)
    
    
    def custom_target_padding1(self, targets, max_len):
        
        zero_targets = torch.zeros(self.config.batch_size, max_len).to(torch.long)
        zero_targets.fill_(self.padding_token)

        for idx, target in enumerate(targets):
            zero_targets[idx].narrow(0,0,len(target)).copy_(torch.LongTensor(target))
        return zero_targets
    
    def shuffle(self, a,b,c,d):
        """ Shuffle dataset """
        tmp = list(zip(a,b,c,d))
        random.shuffle(tmp)
        a,b,c,d = zip(*tmp)
        a = torch.cat(a, dim=0)
        b = torch.cat(b, dim=0).long()
        c = torch.cat(c, dim=0).long()
        d = torch.cat(d, dim=0).long()
        return a,b,c,d
    
    def shuffle_1(self, features, targets, feature_len, target_len, audios, file_path):
        idx = list(range(features.shape[0]))
        random.shuffle(idx)

        new_features = []
        new_targets = []
        new_target_len = []
        new_feature_len = []
        new_audios = []
        new_files = []

        for _idx in idx:
            new_features += [features[_idx].unsqueeze(0)]
            new_targets += [targets[_idx].unsqueeze(0)]
            new_target_len += [target_len[_idx]]
            new_feature_len += [feature_len[_idx]]
            new_audios += [audios[_idx]]
            new_files += [file_path[_idx]]

        features = torch.cat(new_features, dim=0)
        targets = torch.cat(new_targets, dim=0).long()
        target_len = torch.IntTensor(new_target_len)
        feature_len = torch.IntTensor(new_feature_len)
        return features, targets, feature_len, target_len, new_audios, new_files

    def __call__(self, bs):
        '''
        tokenizer.decode(processor.tokenizer.pad([{'input_ids':_a}, {"input_ids":_b}])['input_ids'][0])
        '<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>배우 나영희 윤유선 코미디언 이성미 등도 비통함을 금치 못했다<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>'
        '''
        targets = []
        features = []
        target_len = []
        feature_len = []
        audios = []
        file_fns = []

        for feature, target, audio, file in bs:
            # target = torch.tensor(target).long()
            # feature_len += [min(feature.shape[-1], self.config.mfcc_max_len)]
            feature_len += [feature.shape[-1]]
            target_len += [target.shape[-1]] # baseline에서는 -1을 햇음 왜그랬을까?
            targets += [target]
            audios += [audio]
            file_fns += [file]

        if self.config.pretrained_version == 'whisper':
            max_feature_len = 3000
        else:
            max_feature_len = min(max(feature_len), self.config.mfcc_max_len)
        max_target_len = max(target_len)

        for (feature, target, _, _), current_feature_len, current_target_len in zip(bs, feature_len, target_len):
            #targets += [self.custom_target_padding1(target, max_target_len, current_target_len)]
            features += [self.custom_feature_padding(feature, max_feature_len, current_feature_len).unsqueeze(0)]

        targets = self.custom_target_padding1(targets, max_target_len)
        #targets = torch.cat(targets, dim=0)
        # features = torch.cat(features, dim=0).transpose(-1,-2)
        features = torch.cat(features, dim=0)


        ### 제발 되라...
        final_feature_len = []
        for _f in feature_len:
            if _f > self.config.mfcc_max_len:
                final_feature_len += [self.config.mfcc_max_len]
            else:
                final_feature_len += [_f]
        feature_len = torch.IntTensor(final_feature_len)
        target_len = torch.IntTensor(target_len)

        features, targets, feature_len, target_len, audios, file_fns = self.shuffle_1(features, targets, feature_len, target_len, audios, file_fns)

        return features, targets, feature_len, target_len, audio, file_fns
    

class CustomSpeechSeq2SeqWithPadding():
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, bs):
        input_features = []
        labels = []
        
        for mfcc, ids in bs:
            input_features += [{"input_features" : mfcc}]
            labels += [{"input_ids" : ids}]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt', padding=True)
        labels_batch = self.tokenizer.pad(labels, return_tensors='pt', padding=True)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



class CustomDataset_1(Dataset):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        config (DictConfig): set of configurations
        dataset_path (str): path of dataset
    """

    def __init__(
            self,
            df,
            processor,
            tokenizer,
            config,  # set of arguments
            filename='filename',
            target_col='text',
    ):
        super().__init__()

        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        self.df = df.reset_index(drop=True)
        self.config = config
        self.processor = processor
        self.filename = filename
        self.target_col = target_col
        self.tokenizer = tokenizer

    def wav2image_tensor(self, path):
        audio, sr = librosa.load(path, sr=self.config.sample_rate)
        audio, _ = librosa.effects.trim(audio)

        if self.config.remove_noise:
            audio = remove_noise_data(audio)

        if self.config.del_silence:
            audio = detect_silence(
                audio,
                audio_threshold=self.config.audio_threshold,
                min_silence_len=self.config.min_silence_len,
                ratio = self.config.sample_rate,
                make_silence_len=self.config.make_silence_len
                )


        if not self.valid_yn:
            agument_num = random.randrange(1, 11) # 1~10

            if agument_num <= 2:
                transform = AddGaussianNoise(
                    min_amplitude=0.001,
                    max_amplitude=0.015,
                    p=1.0
                )
                audio = transform(audio, sample_rate=16000)

            elif agument_num <= 4:
                transform = AirAbsorption(
                    min_distance=10.0,
                    max_distance=50.0,
                    p=1.0,
                )
                audio = transform(audio, sample_rate=16000)

            elif agument <= 6:
                transform = AddGaussianSNR(
                    min_snr_db=5.0,
                    max_snr_db=40.0,
                    p=1.0
                )
                audio = transform(audio, sample_rate=16000)
            
            elif agument <= 8 :
                pass # 구현예정  https://github.com/iver56/audiomentations

            elif agument <= 10:         
                pass # 구현예정  https://github.com/iver56/audiomentations
                
        input_features = self.processor(audio, sampling_rate=self.config.sample_rate, return_tensors = 'pt', padding='longest').input_features[0]

        return input_features, audio
    
    
    def freq_masking(self, feat, F = 20, freq_mask_num = 1):
        feat_size = feat.size(1)
        seq_len = feat.size(0)

        # freq mask
        for _ in range(freq_mask_num):
            f = np.random.uniform(low=0.0, high=F)
            f = int(f)
            f0 = random.randint(0, feat_size - f)
            feat[:, f0 : f0 + f] = 0

        return feat

    def time_masking(self, feat, T = 70, time_mask_num = 1):
        feat_size = feat.size(1)
        seq_len = feat.size(0)

        # time mask
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=T)
            t = int(t)
            t0 = random.randint(0, seq_len - t)
            feat[t0 : t0 + t, :] = 0

        return feat


    def __getitem__(self, idx):
        semi_df = self.df.iloc[idx]
        _file_path = semi_df[self.filename]

        mfccs, audio = self.wav2image_tensor(
                _file_path
                )

        # input_len = len(mfccs)

        if self.config.pretrained_version == 'wav2vec':
            with self.processor.as_target_processor():
                label = self.processor(semi_df[self.target_col]).input_ids
        elif self.config.pretrained_version == 'whisper':
            # _label = processor(batch[target_col]).input_ids
            label = self.tokenizer(semi_df[self.target_col], padding=True).input_ids


        if random.randrange(1,11) > 7:
            # _n_filter = np.random.randint(0,1)
            mfccs = self.freq_masking(mfccs, freq_mask_num=1)
        
        if random.randrange(1,11) > 7:
            # _n_filter = np.random.randint(0,2)
            mfccs = self.time_masking(mfccs, time_mask_num=1)


        # sentence_to_chars = []
        # mfccs = []
        # for _file_path, _sentence_to_char in zip(semi_df['filename'], semi_df['sentence_to_char']):

        #     feature = self.wav2image_tensor(
        #         _file_path
        #         )
        #     mfccs += [feature]
        #     sentence_to_chars += [torch.tensor(_sentence_to_char).long()]

        return mfccs, torch.tensor(label).long(), audio, _file_path

    def __len__(self):
        return len(self.df)


class CustomDataset_3(Dataset):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        config (DictConfig): set of configurations
        dataset_path (str): path of dataset
    """

    def __init__(
            self,
            df,
            processor,
            tokenizer,
            config,  # set of arguments
            filename='filename',
            target_col='text',
    ):
        super().__init__()

        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        self.df = df.reset_index(drop=True)
        self.config = config
        self.processor = processor
        self.filename = filename
        self.target_col = target_col
        self.tokenizer = tokenizer

    def wav2image_tensor(self, path):
        audio, sr = librosa.load(path, sr=self.config.sample_rate)
        audio, _ = librosa.effects.trim(audio)

        if self.config.remove_noise:
            audio = remove_noise_data(audio)

        if self.config.del_silence:
            audio = detect_silence(
                audio,
                audio_threshold=self.config.audio_threshold,
                min_silence_len=self.config.min_silence_len,
                ratio = self.config.sample_rate,
                make_silence_len=self.config.make_silence_len
                )

        input_features = self.processor(audio, sampling_rate=self.config.sample_rate, return_tensors = 'pt', padding='longest').input_features[0]

        return input_features, audio
    
    
    def freq_masking(self, feat, F = 20, freq_mask_num = 1):
        feat_size = feat.size(1)
        seq_len = feat.size(0)

        # freq mask
        for _ in range(freq_mask_num):
            f = np.random.uniform(low=0.0, high=F)
            f = int(f)
            f0 = random.randint(0, feat_size - f)
            feat[:, f0 : f0 + f] = 0

        return feat

    def time_masking(self, feat, T = 70, time_mask_num = 1):
        feat_size = feat.size(1)
        seq_len = feat.size(0)

        # time mask
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=T)
            t = int(t)
            t0 = random.randint(0, seq_len - t)
            feat[t0 : t0 + t, :] = 0

        return feat


    def __getitem__(self, idx):
        semi_df = self.df.iloc[idx]
        _file_path = semi_df[self.filename]

        mfccs, audio = self.wav2image_tensor(
                _file_path
                )

        # input_len = len(mfccs)

        if self.config.pretrained_version == 'wav2vec':
            with self.processor.as_target_processor():
                label = self.processor(semi_df[self.target_col]).input_ids
        elif self.config.pretrained_version == 'whisper':
            # _label = processor(batch[target_col]).input_ids
            label = self.tokenizer(semi_df[self.target_col], padding=True).input_ids


        # sentence_to_chars = []
        # mfccs = []
        # for _file_path, _sentence_to_char in zip(semi_df['filename'], semi_df['sentence_to_char']):

        #     feature = self.wav2image_tensor(
        #         _file_path
        #         )
        #     mfccs += [feature]
        #     sentence_to_chars += [torch.tensor(_sentence_to_char).long()]

        return mfccs, torch.tensor(label).long(), audio, _file_path

    def __len__(self):
        return len(self.df)





## @hijung - for inference
class CustomDataset_2(Dataset):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        config (DictConfig): set of configurations
        dataset_path (str): path of dataset
    """

    def __init__(
            self,
            df,
            processor,
            tokenizer,
            config,  # set of arguments
            filename='filename',
            target_col='text',
    ):
        super().__init__()

        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        self.df = df.reset_index(drop=True)
        self.config = config
        self.processor = processor
        self.filename = filename
        self.target_col = target_col
        self.tokenizer = tokenizer

    def wav2image_tensor(self, path):
        audio, sr = librosa.load(path, sr=self.config.sample_rate)
        audio, _ = librosa.effects.trim(audio)
        audio = remove_noise_data(audio)

        
        audio = detect_silence(
            audio,
            audio_threshold=self.config.audio_threshold,
            min_silence_len=self.config.min_silence_len,
            ratio = self.config.sample_rate,
            make_silence_len=self.config.make_silence_len
            )

        input_features = self.processor(audio, sampling_rate=self.config.sample_rate, return_tensors = 'pt', padding='longest').input_features[0]
        return input_features, audio
    

    def __getitem__(self, idx):
        semi_df = self.df.iloc[idx]
        _file_path = semi_df[self.filename]

        mfccs, audio = self.wav2image_tensor(
                _file_path
                )
        # input_len = len(mfccs)
        return mfccs, audio, _file_path

    def __len__(self):
        return len(self.df)