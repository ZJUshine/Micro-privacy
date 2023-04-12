from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import SparkEvaluator
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

import numpy as np
import glob, random, os
from speechbrain.pretrained import SpeakerRecognition
from pystoi import stoi
from pypesq import pesq
import soundfile as sf
from matplotlib import pyplot as plt
import torch
from eval import wer, ASRtrans
from multiprocessing import Pool, Manager
from eval import refer_csv


def moo(label='mopso1',
        random_num = 10,
        n_gen = 50
        ):  ## use STOI


    # define verification
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/home/xsl/micprivacy/pretrained_models/spkrec-ecapa-voxceleb")
    # asr_model = EncoderDecoderASR.from_hparams(source='speechbrain/asr-crdnn-rnnlm-librispeech', savedir='/home/xsl/speechbrain/pretrained_models/asr-crdnn-rnnlm-librispeech')

    # wave_root = '/home/xsl/micprivacy/data/librispeech/train-clean-100-8k/'
    wave_root = '/home/xsl/micprivacy/data/librispeech/test-clean-8k/'
    save_root = '/home/xsl/micprivacy/data/new_test/' + label + '/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    ## randomly load train-set
    wave_paths = glob.glob(os.path.join(wave_root, '*/*/*.wav'))

    # save result to file 
    save_file = '/home/xsl/micprivacy/data/new_test/' + label + '.txt'  # save each iteration result
    # f = open(save_file,'a')


    class Mopso(FloatProblem):

        def __init__(self):
            super().__init__()
            self.number_of_variables = 3
            self.number_of_objectives = 2
            self.number_of_constraints = 0
            self.lower_bound = [0, 0, 0] 
            self.upper_bound = [2, 2, 2]

        def evaluate(self, solution: FloatSolution) -> FloatSolution:
            x = solution.variables

            ## write params to txt file
            f = open('/home/xsl/micprivacy/code/param/param_moo3.txt','w')
            for i in range(3):
                print(x[i], file=f)
            f.close()
        
            ## compile G.729 codec
            os.chdir('/home/xsl/micprivacy/code/g729/c_code_copy3/')
            os.system('rm *.o')
            os.system('make -f coder.mak')
            os.system('make -f decoder.mak')
            os.system('rm *.o')
            # subprocess.run('/home/xsl/micprivacy/code/g729/c_code_copy2/', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run('rm *.o', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run('make -f coder.mak', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run('make -f decoder.mak', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run('rm *.o', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ## calculate obj value
            index = 0
            value1 = 0
            value2 = 0
            train_set = random.sample(wave_paths, random_num) 
            for i in range(random_num):
                orig_path = train_set[index]
                save_path = save_root + os.path.split(orig_path)[1].split('.')[0]
                bin_path = save_path + 'g729.bin'
                pcm_path = save_path + 'g729.pcm'
                wav_path = save_path + 'g729.wav'
                pcm_path_16k = save_path + 'g729_16k.pcm'
                index += 1

                os.system('/home/xsl/micprivacy/code/g729/c_code_copy2/coder %s %s' %(orig_path, bin_path))
                os.system('/home/xsl/micprivacy/code/g729/c_code_copy2/decoder %s %s' %(bin_path, pcm_path))
                # os.system('ffmpeg -f s16le -v 8 -y -ar 8000 -ac 1 -i %s %s' %(pcm_path, wav_path))
                os.system('ffmpeg -y -ar 8000 -ac 1 -f s16le -i %s -ar 16000 -ac 1 -f s16le %s' %(pcm_path, pcm_path_16k))
                os.system('ffmpeg -f s16le -v 8 -y -ar 16000 -ac 1 -i %s %s' %(pcm_path_16k, wav_path))
                # subprocess.run('/home/xsl/micprivacy/code/g729/c_code_copy2/coder %s %s' %(orig_path, bin_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # subprocess.run('/home/xsl/micprivacy/code/g729/c_code_copy2/decoder %s %s' %(bin_path, pcm_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # subprocess.run('ffmpeg -f s16le -v 8 -y -ar 8000 -ac 1 -i %s %s' %(pcm_path, wav_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


                # orig_data, sr = sf.read(orig_path.replace('100-8k','100'))  #注意这里用16k音频评估
                orig_data, sr = sf.read(orig_path.replace('-8k',''))  #注意这里用16k音频评估
                anon_data, sr = sf.read(wav_path)

                # calculate ASV cosine distance
                score_asv, _ = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))  # 0~1
                score_asv = float(score_asv[0])
                value1 += score_asv

                min_len = np.min([len(orig_data), len(anon_data)])
                # get stoi
                score_stoi = stoi(orig_data[0:min_len], anon_data[0:min_len], sr)  # 0-1
                value2 += 1-score_stoi

                # calculate loss
                f = open(save_file,'a')
                print("%d wave_path:%s ; asv_score:%.6f ; stoi:%.6f; value1:%.6f; value2:%.6f" %(index, orig_path, score_asv, score_stoi, value1, value2), file=f)
                print(x , file=f)
                # print('-'*20, file=f)
                f.close()

            solution.objectives = [value1, value2]
            return solution
        
        def get_name(self):
            return label
        
    problem = Mopso()

    algorithm = OMOPSO(
        problem=problem,
        swarm_size=50,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=1.0/3, perturbation=0.5),  # probability = 1/number_of_variables
        non_uniform_mutation=NonUniformMutation(
            1.0/3, perturbation=0.5, max_iterations = n_gen
        ),  # max_iterations=max_evaluations / swarm_size
        leaders=CrowdingDistanceArchive(10),
        termination_criterion=StoppingByEvaluations(max_evaluations=n_gen*50),
    )

    algorithm.run()
    front = algorithm.get_result()

    # save results to file
    print_function_values_to_file(front, '/home/xsl/micprivacy/data/new_test/FUN.' + problem.get_name())
    print_variables_to_file(front, '/home/xsl/micprivacy/data/new_test/VAR.' + problem.get_name())

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")

if __name__ == '__main__':
    moo(label='mopso1')