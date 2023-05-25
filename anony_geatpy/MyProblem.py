# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from CELP import CELP
import soundfile as sf
import fastwer
import pandas as pd
import pystoi
import torch
import os


from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../../pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderASR
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../../pretrained_models/asr-wav2vec2-librispeech")



class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 10  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-0.2] * Dim  # 决策变量下界
        ub = [0.2] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):  # 目标函数
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        x6 = Vars[:, [5]]
        x7 = Vars[:, [6]]
        x8 = Vars[:, [7]]
        x9 = Vars[:, [8]]
        x10 = Vars[:, [9]]
        df = pd.read_table('../anony_jMetalPy/tools/test-clean-trans.txt', sep=',')
        # 随机抽取10条音频
        random_rows = df.sample(n=5).values
        scores_all = [];stoi_values_all = [];wer_values_all = []
        for i in range(len(x1)):
            x = [x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i]]
            scores = [];stoi_values = [];wer_values = []
            for random_row in random_rows:
                audio_name = random_row[0]
                ref = [random_row[1]]
                path0 = audio_name.split("-")[0]
                path1 = audio_name.split("-")[1]
                wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
                codec = CELP(wave_path = wave_path,save_path=f'./results/anony_audio/{audio_name}.flac',anonypara=x,anony=True)
                _,lsfs,modif_lsfs = codec.run()
                orig_data, sr = sf.read(wave_path)
                anon_data, sr = sf.read(f'./results/anony_audio/{audio_name}.flac')
                score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))
                stoi_value = pystoi.stoi(orig_data, anon_data, sr, extended=False)
                asr_result = asr_model.transcribe_file(f'./results/anony_audio/{audio_name}.flac')
                wer = fastwer.score([asr_result], ref)
                scores.append(score.numpy())
                stoi_values.append(1-stoi_value)
                wer_values.append(wer)
                os.remove(f'./results/anony_audio/{audio_name}.flac')
                os.remove(f'./{audio_name}.flac')
            score_mark = sum(scores)/len(scores)
            score_mark = score_mark[0][0]
            stoi_value_mark = sum(stoi_values)/len(stoi_values)
            wer_mark = sum(wer_values)/len(wer_values)
            scores_all.append([score_mark])
            stoi_values_all.append([stoi_value_mark])
            wer_values_all.append([wer_mark])
            print(f"score:{score_mark},stio:{stoi_value_mark},wer:{wer_mark}")

        f1 = scores_all
        f2 = wer_values_all
        print(f1)
        print(f2)
        f = np.hstack([f1, f2])
        return f