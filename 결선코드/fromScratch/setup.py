#!nova: whmrtm/wav2vecrc:latest
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
        'torchaudio',
        'pydub',
        'glob2',
        'transformers >= 4.3.0',
        'evaluate',
        'jiwer',
        'audiomentations',
        'pyctcdecode==0.5.0',
        'pypi-kenlm==0.1.20220713'
    ],
)

