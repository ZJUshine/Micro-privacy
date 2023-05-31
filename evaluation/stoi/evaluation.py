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
from speechbrain.pretrained import EncoderASR
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../../pretrained_models/asr-wav2vec2-librispeech")


df = pd.read_table('../../anony_jMetalPy/tools/test-clean-trans.txt', sep=',')
rows = df.values

x_stoi_0 = [1.768681729503927130e-02,-7.771053150393513603e-03,6.352773653774629325e-02,-7.804047768264515961e-02,\
            1.209184337801006565e-01,1.361366979216110419e-02,1.857338175180986428e-01,-1.246755640953779026e-01,\
                1.203702033963054696e-01,-2.369460992049425840e-01]
x_stoi_1 = [1.772716248938305608e-02,-7.771053150393513603e-03,6.225459854653631747e-02,-7.195391795834056581e-02,\
            1.209184337801006426e-01,1.361366979216110419e-02,1.857338175180986428e-01,-1.246755640953778888e-01,\
                1.203702033963054696e-01,-2.389088504898624898e-01]
x_stoi_best = [-1.482047694059791254e-01,-9.140567503444804731e-02,-5.724058079446146807e-02,7.678833410825808281e-02,\
                2.437734419272227088e-02,5.610054228670299098e-02,-1.480385760299656184e-01,3.935628246569944366e-02,\
                    1.838913893783652165e-01,-1.883972729983803551e-01]
x_stoi_list = [x_stoi_best]
for i,x in enumerate(x_stoi_list):
    scores = [];stoi_values = [];wer_values = []
    for row in rows:
        audio_name = row[0]
        ref = [row[1]]
        path0 = audio_name.split("-")[0]
        path1 = audio_name.split("-")[1]
        wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
        codec = CELP(wave_path = wave_path,save_path=f'/mnt/lxc/Micro-privacy_data/eval/stoi/best/{audio_name}.flac',anonypara=x)
        _,lsfs,modif_lsfs = codec.run()
        orig_data, sr = sf.read(wave_path)
        code_data, sr = sf.read(f'/mnt/lxc/Micro-privacy_data/eval/stoi/best/{audio_name}.flac')
        score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(code_data))
        stoi_value = pystoi.stoi(orig_data, code_data, sr, extended=False)
        asr_result = asr_model.transcribe_file(f'/mnt/lxc/Micro-privacy_data/eval/stoi/best/{audio_name}.flac')
        wer = fastwer.score([asr_result], ref)
        print(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}")
        with open(f'evaluation_stoi_best_{i}.txt', "a") as f:
            f.write(f"name:{audio_name},score:{score},stoi:{stoi_value},wer:{wer}\n")
        scores.append(score.numpy())
        stoi_values.append(1-stoi_value)
        wer_values.append(wer)
        os.remove(f'./{audio_name}.flac')

    with open(f'evaluation_stoi_best_{i}.txt', "a") as f:
        f.write(f"average score:{np.mean(scores)},stoi:{np.mean(stoi_values)},wer:{np.mean(wer_values)}\n")
        f.write(f"min score:{np.min(scores)},stoi:{np.min(stoi_values)},wer:{np.min(wer_values)}\n")
        f.write(f"max score:{np.max(scores)},stoi:{np.max(stoi_values)},wer:{np.max(wer_values)}\n")