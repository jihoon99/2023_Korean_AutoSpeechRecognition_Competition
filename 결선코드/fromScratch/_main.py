import torch
import os
import warnings
import time
import argparse
# from glob import glob # glob이 도커에 안깔림.
# import random
# import json
# import queue

from modules.preprocess import preprocessing
from modules.trainer import trainer
from modules.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
# from modules.audio import (
#     FilterBankConfig,
#     MelSpectrogramConfig,
#     MfccConfig,
#     SpectrogramConfig,
# )
# from modules.model import build_model
from modules.model.deepspeech2 import build_deepspeech2
from modules.vocab import KoreanSpeechVocabulary
from modules.data import split_dataset, collate_fn
from modules.utils import Optimizer
from modules.metrics import get_metric
from modules.inference import single_infer

from torch.utils.data import DataLoader
import torch.nn as nn

from modules.vocab import Vocabulary

import pandas as pd

import nova
from nova import DATASET_PATH


from dotenv import load_dotenv # read '.env' file
load_dotenv()



def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.



def inference(path, model, **kwargs):
    model.eval()

    results = []
    for i in [os.path.join(path,i) for i in os.listidr(path)]:
    # for i in glob(os.path.join(path, '*')): # glob이 도커에 안깔림...
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])

def spell_check(config):
    pass

def load_vocab(
        config,
        sos_id = '<s>',
        eos_id = '</s>',
        pad_id = '[pad]',
        blank_id = '<blank>',
        # unk_id = '[unk]',
        out_path = 'labels.csv'
    ):
    vocab = KoreanSpeechVocabulary(
        os.path.join(os.getcwd(), out_path), 
        output_unit='character',
        sos_id = sos_id,
        eos_id = eos_id,
        pad_id = pad_id,
        blank_id = blank_id,
        # unk_id = unk_id
        )
        
    return vocab # class it self


def build_model(
        input_size,
        config,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:

    if config.architecture == 'deepspeech2':

        model = build_deepspeech2(
            input_size    = input_size,
            num_classes   = len(vocab),
            rnn_type      = config.rnn_type,
            num_rnn_layers= config.num_encoder_layers,
            rnn_hidden_dim= config.hidden_dim,
            dropout_p     = config.dropout,
            bidirectional = config.use_bidirectional,
            activation    = config.activation,
            device        = device,
        )

    return model



if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')   # mode - related to nova system. barely change
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)



    # Parameters 

    args.add_argument('--version', type=str, default='POC')
    args.add_argument('--make_bow', type=bool, default=True)

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=128)

    args.add_argument('--save_result_every', type=int, default=2) # 2 
    args.add_argument('--checkpoint_every', type=int, default=1)  # save model every 
    args.add_argument('--print_every', type=int, default=50)
    
    args.add_argument('--dataset', type=str, default='kspon')            # check
    args.add_argument('--output_unit', type=str, default='character')    # check

    # Data Processing
    args.add_argument('--audio_extension', type=str, default='wav')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    # DataLoader    
    args.add_argument('--num_workers', type=int, default=16)
    args.add_argument('--num_threads', type=int, default=16)
    
    # Optimizer lr scheduler options
    args.add_argument("--constant_lr", default=5e-5)  # when this is False:   아래의 옵션들이 실행됨.
    args.add_argument('--use_lr_scheduler', type=bool, default=False)
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--warmup_steps', type=int, default=1000)
    args.add_argument('--weight_decay', type=float, default=1e-05)

    # optimizer
    args.add_argument('--optimizer', type=str, default='adam')       

    # explode 예방
    args.add_argument('--max_grad_norm', type=int, default=400)

    # check
    args.add_argument('--reduction', type=str, default='mean')        # check
    args.add_argument('--total_steps', type=int, default=200000)


    args.add_argument('--architecture', type=str, default='deepspeech2') # check
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')           # check
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--use_bidirectional', type=bool, default=True) # maybe decoder aggregator


    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)  # check
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)   # check
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)  # check
    args.add_argument('--joint_ctc_attention', type=bool, default=False)   # check
    config = args.parse_args()

    warnings.filterwarnings('ignore')

    # seed
    # random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # gpu
    device = 'cuda' if config.use_cuda == True else 'cpu'
    
    # thread
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    # vocab 사전 가지고와 / 사전 csv 파일 열어서, pad, sos, eos, blank_id, unk_id 바꿔.
    vocab = load_vocab(
        config,
        sos_id = '<s>',
        eos_id = '</s>',
        pad_id = '[pad]',
        blank_id = '<blank>',
        # unk_id = '[unk]'
    )

    print('load vocab success')
    print('-'*100)
    print('building model')

    model = build_model(
        input_size=config.n_mels,
        config=config, 
        vocab=vocab, 
        device=device)

    print(model)
    print('build model success')
    print("-"*100)

    # load optimizer
    optimizer = get_optimizer(model, config)
    bind_model(model, optimizer=optimizer)
    metric = get_metric(metric_name='CER', vocab=vocab) # this must be edit

    if config.pause:
        nova.paused(scope=locals())

    if config.mode == 'train':
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data') # 트레이닝 데이터 셋 위치 
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')         # 정답지 위치 : train_label이 csv인가 봅니다. -> columns : audio_path, transcript(sentence)
        preprocessing(label_path, os.getcwd()) # current_path/label.csv 가 존재합니다. (git에도 제가 일부를 올렸음.) # transcript

        # config.version == 'PoC' 일 경우, 일부 데이터만 갖고 train_dataset, valid_datset 구성됨.
        train_dataset, valid_dataset = split_dataset(config, os.path.join(os.getcwd(), 'transcripts.txt'), vocab) # data 부분에서 무음 처리 하는 부분과 broadcasting 부분 바꿔야함.

        # lr 스케쥴 적용한 것과 아닌것.
        if config.constant_lr == False:
            lr_scheduler = get_lr_scheduler(config, optimizer, len(train_dataset)) # learning scheduler 적용했네.
            optimizer = Optimizer(optimizer, lr_scheduler, int(len(train_dataset)*config.num_epochs), config.max_grad_norm)

        criterion = get_criterion(config, vocab) # CTC loss


        train_begin_time = time.time()


        for epoch in range(config.num_epochs):
            print('[INFO] Epoch %d start' % epoch)

            # train

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,            # 배치 내에서 max값에 padding을 해줬는데, for 사용함 : 속도 느림. torch.pad(?) 사용하자. 적극적으로 broadcasting사용 -> rainism repository : asfl kaggle : 참고
                num_workers=config.num_workers
            )

            model, train_loss, train_cer = trainer(
                'train',
                config,
                train_loader,
                optimizer,
                model,
                criterion,
                metric,
                train_begin_time,
                device
            )

            print('[INFO] Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

            # valid

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config.num_workers
            )

            model, valid_loss, valid_cer = trainer(
                'valid',
                config,
                valid_loader,
                optimizer,
                model,
                criterion,
                metric,
                train_begin_time,
                device
            )

            print('[INFO] Epoch %d (Validation) Loss %0.4f  CER %0.4f' % (epoch, valid_loss, valid_cer))

            nova.report(
                summary=True,
                epoch=epoch,
                train_loss=train_loss,
                train_cer=train_cer,
                val_loss=valid_loss,
                val_cer=valid_cer
            )

            if epoch % config.checkpoint_every == 0:
                nova.save(epoch)

            torch.cuda.empty_cache()
            print(f'[INFO] epoch {epoch} is done')
        print('[INFO] train process is done')