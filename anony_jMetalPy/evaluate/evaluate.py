from CELP import CELP
import soundfile as sf
import fastwer
import pystoi
import torch
import pandas as pd
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../../pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderASR
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../../pretrained_models/asr-wav2vec2-librispeech")

x = [0.17024790226534792,1.141644026608388,1.1320157822676724,-0.012825904550725978]
df = pd.read_table('../tools/test-clean-trans-all.txt', sep=',')
scores = [];stoi_values = [];wer_values = []
for audio_row in df.values:
    audio_name = audio_row[0]
    ref = [audio_row[1]]
    path0 = audio_name.split("-")[0]
    path1 = audio_name.split("-")[1]
    wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
    codec = CELP(wave_path = wave_path,save_path=f'/mnt/lxc/anony_audio/{audio_name}.flac',anonypara=x,anony=True)
    _,lsfs,modif_lsfs = codec.run()
    print(f"lsfs:{lsfs[0]}")
    print(f"modif_lsfs:{modif_lsfs[0]}")
    orig_data, sr = sf.read(wave_path)
    anon_data, sr = sf.read(f'/mnt/lxc/anony_audio/{audio_name}.flac')
    score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))
    stoi_value = pystoi.stoi(orig_data, anon_data, sr, extended=False)
    asr_result = asr_model.transcribe_file(f'/mnt/lxc/anony_audio/{audio_name}.flac')
    wer = fastwer.score([asr_result], ref)
    scores.append(score)
    stoi_values.append(1-stoi_value)
    wer_values.append(wer)
    os.remove(f'/mnt/lxc/anony_audio/{audio_name}.flac')
    os.remove(f'./{audio_name}.flac')
score_mark = sum(scores)/len(scores)
stoi_value_mark = sum(stoi_values)/len(stoi_values)
wer_mark = sum(wer_values)/len(wer_values)
print(f"score_mark:{score_mark},stoi_value_mark:{stoi_value_mark},wer_mark:{wer_mark}")