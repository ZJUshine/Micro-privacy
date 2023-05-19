from CELP import CELP
import soundfile as sf
import fastwer
import pandas as pd
import pystoi
import torch
import os
import numpy as np
# 导入speechbrain预训练模型
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderASR
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../pretrained_models/asr-wav2vec2-librispeech")


df = pd.read_table('../anony_jMetalPy/tools/test-clean-trans.txt', sep=',')
rows = df.values
scores = [];stoi_values = [];wer_values = []
for row in rows:
    audio_name = row[0]
    ref = [row[1]]
    path0 = audio_name.split("-")[0]
    path1 = audio_name.split("-")[1]
    wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
    codec = CELP(wave_path = wave_path,save_path=f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac')
    _,lsfs = codec.run()
    orig_data, sr = sf.read(wave_path)
    code_data, sr = sf.read(f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac')
    score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(code_data))
    stoi_value = pystoi.stoi(orig_data, code_data, sr, extended=False)
    asr_result = asr_model.transcribe_file(f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac')
    wer = fastwer.score([asr_result], ref)
    print(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}")
    with open(f'CELP_influence_data.txt', "a") as f:
        f.write(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}\n")
    scores.append(score)
    stoi_values.append(1-stoi_value)
    wer_values.append(wer)
    os.remove(f'./{audio_name}.flac')

with open(f'CELP_influence_data.txt', "a") as f:
    f.write(f"average score:{np.mean(scores)},stoi:{np.mean(stoi_values)},wer:{np.mean(wer_values)}\n")
    f.write(f"min score:{np.min(scores)},stoi:{np.min(stoi_values)},wer:{np.min(wer_values)}\n")
    f.write(f"max score:{np.max(scores)},stoi:{np.max(stoi_values)},wer:{np.max(wer_values)}\n")