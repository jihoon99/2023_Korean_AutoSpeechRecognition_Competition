#!nova: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
from distutils.core import setup

setup(
    name='koreanAI2023',
    version='latest',
    install_requires=[
        # 'torch==1.7.0',
        # 'levenshtein',
        'noisereduce',
        'librosa >= 0.7.0',
        'numpy==1.21.6',
        'scipy',
        'pandas',
        'tqdm',
        'matplotlib',
        'astropy',
        'sentencepiece',
        'torchaudio==0.6.0',
        'pydub',
        'glob2',
        'transformers',
        'evaluate',
        'jiwer',
        'audiomentations',
        'pyctcdecode==0.5.0',
        #'pypi-kenlm'
    ],
)

