'''
FilePath: define_moead_problem.py
Author: zjushine
Date: 2023-05-03 13:59:18
LastEditors: zjushine
LastEditTime: 2023-05-06 17:08:38
Description: 定义一个moead问题
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
import os
from datetime import datetime
import wandb
# 导入speechbrain预训练模型
# from speechbrain.pretrained import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="../../pretrained_models/spkrec-xvect-voxceleb")
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../../pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderASR
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../../pretrained_models/asr-wav2vec2-librispeech")
sim_cal = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
_date = '{}'.format(datetime.now().strftime("%m%d"))
now = '{}'.format(datetime.now().strftime("%H%M"))

results_output_path = f"results/{_date}_{now}"
class moead(FloatProblem):
    """
    opt_target : 
        Type : str by default None
        Describe : 优化目标，可选项为：'stoi' 'wer' 'both'
    """
    def __init__(self,opt_target = None):
        super().__init__()
        self.opt_target = opt_target
        self.number_of_variables = 10
        # self.number_of_variables = 3
        if self.opt_target == 'both':
            self.number_of_objectives = 3
        else:
            self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.lower_bound = [-0.05,-0.05,-0.1,-0.1,-0.15,-0.15,-0.2,-0.2,-0.25,-0.25] 
        self.upper_bound = [0.05,0.05,0.1,0.1,0.15,0.15,0.2,0.2,0.25,0.25]
        self.epoch = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        self.epoch += 1
        # x 为需要优化的参数
        x = solution.variables
        df = pd.read_table('../tools/test-clean-trans.txt', sep=',')
        # 随机抽取10条音频
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
            # print(f"modif_lsfs:{modif_lsfs}")
            orig_data, sr = sf.read(wave_path)
            anon_data, sr = sf.read(f'./results/anony_audio/{audio_name}.flac')
            # embeddings1 = classifier.encode_batch(torch.tensor(orig_data))
            # embeddings2 = classifier.encode_batch(torch.tensor(anon_data))
            # score = sim_cal(embeddings1, embeddings2)
            score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))
            stoi_value = pystoi.stoi(orig_data, anon_data, sr, extended=False)
            asr_result = asr_model.transcribe_file(f'./results/anony_audio/{audio_name}.flac')
            wer = fastwer.score([asr_result], ref)
            # print(f"asr_result:{asr_result},ref:{ref}")
            # print(f"x:{x},score:{score},stio:{stoi_value},wer:{wer}")
            scores.append(score)
            stoi_values.append(1-stoi_value)
            wer_values.append(wer)
            os.remove(f'./results/anony_audio/{audio_name}.flac')
            os.remove(f'./{audio_name}.flac')
        score_mark = sum(scores)/len(scores)
        stoi_value_mark = sum(stoi_values)/len(stoi_values)
        wer_mark = sum(wer_values)/len(wer_values)
        if self.opt_target == "both":
            solution.objectives = [score_mark, stoi_value_mark,wer_mark]
        else:
            if self.opt_target == "stoi":
                solution.objectives = [score_mark,stoi_value_mark]
            else:
                solution.objectives = [score_mark,wer_mark]
        print(f"epoch:{self.epoch},score:{score_mark},stio:{stoi_value_mark},wer:{wer_mark}")

        with open(f'{results_output_path}/data.txt', "a") as f:
                f.write(f"epoch:{self.epoch},score:{score_mark},stio:{stoi_value_mark},wer:{wer_mark},x:{x}\n")
        wandb.log({"score": score_mark, "stio": stoi_value_mark,"wer":wer_mark})
        return solution
    
    def get_name(self):
        return "moead"