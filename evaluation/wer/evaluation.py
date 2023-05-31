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
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../../pretrained_models/spkrec-ecapa-voxceleb")
# from speechbrain.pretrained import EncoderASR
# asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../../pretrained_models/asr-wav2vec2-librispeech")

from speechbrain.pretrained import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="../../pretrained_models/asr-crdnn-rnnlm-librispeech")

df = pd.read_table('../../anony_jMetalPy/tools/test-clean-trans-wer.txt', sep=',')
rows = df.values

x_wer_0 = [-2.051060100256165905e-02,1.868469258204319416e-02,-1.033536037095581818e-02,7.895556476526205403e-02,\
1.082179264615995389e-02,8.754378824390215974e-02,1.390487721189856418e-01,1.786383073776960095e-01,\
8.572017773985862732e-03,1.089513754007719742e-01]
x_wer_1 = [-2.051060100256165905e-02,2.680153889492590397e-02,-1.033536037095581818e-02,7.895556476526208178e-02,\
1.139520190821841604e-02,8.754378824390215974e-02,1.390487721189856696e-01,1.786383073776960095e-01,\
1.331730514795892266e-02,1.089513754007719742e-01]
x_wer_2 = [-2.051060100256166252e-02,1.855948723172299100e-02,-1.033536037095581991e-02,7.895556476526208178e-02,\
1.139520190821842124e-02,1.243366364916781430e-01,1.723048878335843026e-01,1.786383073776959818e-01,\
8.572017773985862732e-03,1.089513754007719604e-01]
x_wer_3 = [-2.051060100256165905e-02,2.680153889492590744e-02,-1.033536037095581818e-02,7.895556476526208178e-02,\
1.139520190821841604e-02,8.754378824390215974e-02,1.390487721189856696e-01,1.786383073776960095e-01,\
1.331730514795892266e-02,1.089513754007719742e-01]
x_wer_list = [x_wer_0,x_wer_1,x_wer_2,x_wer_3]
for i,x in enumerate(x_wer_list):
    scores = [];stoi_values = [];wer_values = [];mmr = []
    for row in rows:
        audio_name = row[0]
        ref = [row[1]]
        path0 = audio_name.split("-")[0]
        path1 = audio_name.split("-")[1]
        wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
        codec = CELP(wave_path = wave_path,save_path=f'/mnt/lxc/Micro-privacy_data/eval/wer/{audio_name}.flac',anonypara=x)
        _,lsfs,modif_lsfs = codec.run()
        orig_data, sr = sf.read(wave_path)
        eval_data, sr = sf.read(f'/mnt/lxc/Micro-privacy_data/eval/wer/{audio_name}.flac')
        score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(eval_data))
        stoi_value = pystoi.stoi(orig_data, eval_data, sr, extended=False)
        asr_result = asr_model.transcribe_file(f'/mnt/lxc/Micro-privacy_data/eval/wer/{audio_name}.flac')
        wer = fastwer.score([asr_result], ref)
        print(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}")
        with open(f'evaluation_wer_crdnn_{i}.txt', "a") as f:
            f.write(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}\n")
        scores.append(score.numpy())
        if score.numpy() < 0.25:
            mmr.append(1)
        else:
            mmr.append(0)
        stoi_values.append(1-stoi_value)
        wer_values.append(wer)
        os.remove(f'./{audio_name}.flac')

    with open(f'evaluation_wer_crdnn_{i}.txt', "a") as f:
        f.write(f"average score:{np.mean(scores)},stoi:{np.mean(stoi_values)},wer:{np.mean(wer_values)}\n")
        f.write(f"min score:{np.min(scores)},stoi:{np.min(stoi_values)},wer:{np.min(wer_values)}\n")
        f.write(f"max score:{np.max(scores)},stoi:{np.max(stoi_values)},wer:{np.max(wer_values)}\n")
        f.write(f"mmr:{np.mean(mmr)}\n")

# x_stoi_0 = [1.768681729503927130e-02,-7.771053150393513603e-03,6.352773653774629325e-02,-7.804047768264515961e-02,\
#             1.209184337801006565e-01,1.361366979216110419e-02,1.857338175180986428e-01,-1.246755640953779026e-01,\
#                 1.203702033963054696e-01,-2.369460992049425840e-01]
# x_stoi_1 = [1.772716248938305608e-02,-7.771053150393513603e-03,6.225459854653631747e-02,-7.195391795834056581e-02,\
#             1.209184337801006426e-01,1.361366979216110419e-02,1.857338175180986428e-01,-1.246755640953778888e-01,\
#                 1.203702033963054696e-01,-2.389088504898624898e-01]
# x_stoi_list = [x_stoi_0,x_stoi_1]
# for i,x in enumerate(x_stoi_list):
#     scores = [];stoi_values = [];wer_values = []
#     for row in rows:
#         audio_name = row[0]
#         ref = [row[1]]
#         path0 = audio_name.split("-")[0]
#         path1 = audio_name.split("-")[1]
#         wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
#         codec = CELP(wave_path = wave_path,save_path=f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac',anonypara=x)
#         _,lsfs,modif_lsfs = codec.run()
#         orig_data, sr = sf.read(wave_path)
#         code_data, sr = sf.read(f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac')
#         score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(code_data))
#         stoi_value = pystoi.stoi(orig_data, code_data, sr, extended=False)
#         asr_result = asr_model.transcribe_file(f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac')
#         wer = fastwer.score([asr_result], ref)
#         print(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}")
#         with open(f'evaluation_stoi_{i}.txt', "a") as f:
#             f.write(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}\n")
#         scores.append(score.numpy())
#         stoi_values.append(1-stoi_value)
#         wer_values.append(wer)
#         os.remove(f'./{audio_name}.flac')

#     with open(f'evaluation_stoi_{i}.txt', "a") as f:
#         f.write(f"average score:{np.mean(scores)},stoi:{np.mean(stoi_values)},wer:{np.mean(wer_values)}\n")
#         f.write(f"min score:{np.min(scores)},stoi:{np.min(stoi_values)},wer:{np.min(wer_values)}\n")
#         f.write(f"max score:{np.max(scores)},stoi:{np.max(stoi_values)},wer:{np.max(wer_values)}\n")