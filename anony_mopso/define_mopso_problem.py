'''
FilePath: define_mopso_problem.py
Author: zjushine
Date: 2023-04-07 14:02:33
LastEditors: zjushine
LastEditTime: 2023-04-25 15:38:05
Description: 定义一个MOPSO的优化问题
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from CELP import CELP
import soundfile as sf
import fastwer
import pystoi

from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/home/lxc/zero/speechbrain/pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="/home/lxc/zero/speechbrain/pretrained_models/asr-crdnn-rnnlm-librispeech")
import torch

class omopso(FloatProblem):

    def __init__(self):
        super().__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 3
        self.number_of_constraints = 0
        self.lower_bound = [-0.1, 0.9] 
        self.upper_bound = [0.1, 1.1]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        orig_data, sr = sf.read('61-70968-0000.flac')
        anon_data, sr = sf.read('61-70968-0000test.flac')
        codec = CELP(wave_path=r'61-70968-0000.flac',
                        save_path=r'61-70968-0000test.flac',anonypara=x,anony=True)
        ref = ["HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT"]
        _,lsfs = codec.run()
        print(f"lsfs:{lsfs}")
        score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))
        stoi_value = pystoi.stoi(orig_data, anon_data, sr, extended=False)
        asr_result = asr_model.transcribe_file("61-70968-0000test.flac")
        
        wer = fastwer.score([asr_result], ref)
        print(f"x:{x},score:{score},stio:{stoi_value},wer:{wer}")
        print(asr_result)
        solution.objectives = [score, 1-stoi_value,wer]
        return solution
    
    def get_name(self):
        return "test"