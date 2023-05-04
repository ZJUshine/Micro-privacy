'''
FilePath: define_moeda_problem.py
Author: zjushine
Date: 2023-05-03 13:59:18
LastEditors: zjushine
LastEditTime: 2023-05-04 15:19:58
Description: 定义一个moeda问题
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from CELP import CELP
import soundfile as sf
import fastwer
import pandas as pd
import pystoi
import torch
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../speechbrain/pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../speechbrain/pretrained_models/asr-crdnn-rnnlm-librispeech")

class moeda(FloatProblem):

    def __init__(self):
        super().__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 3
        self.number_of_constraints = 0
        self.lower_bound = [0, 0.8,0.9,-0.1] 
        self.upper_bound = [0.2, 1.2,1.1,0.1]
        self.epoch = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        self.epoch += 1
        x = solution.variables
        df = pd.read_table('./tools/test-clean-trans.txt', sep=',')
        random_rows = df.sample(n=10).values
        scores = [];stoi_values = [];wer_values = []
        for random_row in random_rows:
            audio_name = random_row[0]
            ref = [random_row[1]]
            path0 = audio_name.split("-")[0]
            path1 = audio_name.split("-")[1]
            wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
            codec = CELP(wave_path = wave_path,save_path=f'./results/anony_audio/{audio_name}.flac',anonypara=x,anony=True)
            _,lsfs,modif_lsfs = codec.run()
            print(f"modif_lsfs:{modif_lsfs}")
            orig_data, sr = sf.read(wave_path)
            anon_data, sr = sf.read(f'./results/anony_audio/{audio_name}.flac')
            score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))
            stoi_value = pystoi.stoi(orig_data, anon_data, sr, extended=False)
            asr_result = asr_model.transcribe_file(f'./results/anony_audio/{audio_name}.flac')
            wer = fastwer.score([asr_result], ref)
            # print(f"asr_result:{asr_result},ref:{ref}")
            # print(f"x:{x},score:{score},stio:{stoi_value},wer:{wer}")
            scores.append(score)
            stoi_values.append(1-stoi_value)
            wer_values.append(wer)
        score_mark = sum(scores)/len(scores)
        stoi_value_mark = sum(stoi_values)/len(stoi_values)
        wer_mark = sum(wer_values)/len(wer_values)
        solution.objectives = [score_mark, stoi_value_mark,wer_mark]
        print(f"epoch:{self.epoch},score:{score_mark},stio:{stoi_value_mark},wer:{wer_mark}")
        return solution
    
    def get_name(self):
        return "test"