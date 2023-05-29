from CELP import CELP
import soundfile as sf
import pandas as pd
import torch
import os
import numpy as np
import glob
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="../../pretrained_models/spkrec-xvect-voxceleb")
sim_cal = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


path = '/mnt/lxc/librispeech/test-clean/test-clean-audio'

all_speakers_path = sorted(glob.glob(path+'/*'))
all_speakers = []
for speakers in all_speakers_path:
    all_speakers.append(speakers.split('/')[-1])
all_speakers = all_speakers[:10]
all_files = []
for speaker in all_speakers:
    temp_files = sorted(glob.glob(path+'/'+speaker+'/*/*.flac')[:4])
    for i in temp_files:
        all_files.append(i)

df_trial = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
for i, file in tqdm(enumerate(all_files)):
    all_speakers_temp = all_speakers.copy()
    cur_spk = file.split('/')[-3]
    try:
        all_speakers_temp.remove(cur_spk)
    except:
        pass
    for i in range (2):
        test_spk = random.choice(all_speakers_temp)
        test_file = random.choice(glob.glob(path+'/'+test_spk+'/*/*.flac'))
        is_target = 0
        df_trial = df_trial.append({"target": is_target, "wav1": file, "wav2": test_file},ignore_index=True)
    for i in range (2):
        test_file = random.choice(glob.glob(path+'/'+cur_spk+'/*/*.flac'))
        is_target = 1
        df_trial = df_trial.append({"target": is_target, "wav1": file, "wav2": test_file},ignore_index=True)
print(df_trial)
df_trial.to_csv('./df_trial.txt', sep=' ',index=False)