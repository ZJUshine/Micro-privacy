'''
FilePath: define_mopso_problem.py
Author: zjushine
Date: 2023-04-07 14:02:33
LastEditors: zjushine
LastEditTime: 2023-04-26 14:36:33
Description: 定义一个MOPSO的优化问题
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from CELP import CELP
import soundfile as sf
import fastwer
import pandas as pd

from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/home/lxc/zero/speechbrain/pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="/home/lxc/zero/speechbrain/pretrained_models/asr-crdnn-rnnlm-librispeech")
import torch

class omopso(FloatProblem):

    def __init__(self):
        super().__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.lower_bound = [-0.5, 0.5,0.9,-0.1] 
        self.upper_bound = [0.5, 1.5,1.1,0.1]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        df = pd.read_table('/home/lxc/zero/Micro-privacy/anony_mopso/tools/test-clean-trans.txt', sep=',')
        random_row = df.sample(n=1)
        audio_name = random_row.values[0][0]
        ref = [random_row.values[0][1]]
        
        path0 = audio_name.split("-")[0]
        path1 = audio_name.split("-")[1]
        wave_path = f"/mnt/lxc/librispeech/test-clean/test-clean-audio/{path0}/{path1}/{audio_name}.flac"
        codec = CELP(wave_path = wave_path,
                        save_path=f'./results/anony_audio/{audio_name}.flac',anonypara=x,anony=True)
        _,lsfs = codec.run()
        orig_data, sr = sf.read(wave_path)
        anon_data, sr = sf.read(f'./results/anony_audio/{audio_name}.flac')
        # print(f"lsfs:{lsfs}")
        score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))
        asr_result = asr_model.transcribe_file(f'./results/anony_audio/{audio_name}.flac')
        print(f"asr_result:{asr_result},ref:{ref}")
        wer = fastwer.score([asr_result], ref)
        print(f"x:{x},score:{score},wer:{wer}")
        solution.objectives = [score,wer]
        return solution
    
    def get_name(self):
        return "test"