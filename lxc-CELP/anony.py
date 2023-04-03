from CELP import CELP
import sys
sys.path.append('/home/lxc/zero/speechbrain/pretrained_models')
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
from sko.PSO import PSO


def demo_func(x):
    codec = CELP(wave_path=r'/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac',
                    save_path=r'61-70968-0000test.flac',anonypara=x)
    
    _,lsfs = codec.run()
    score, prediction = verification.verify_files("61-70968-0000.flac", "61-70968-0000test.flac")
    print(x,score)
    
    return score

pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=5, lb=[0, 0], ub=[1, 1], w=0.8, c1=2, c2=2)
pso.run(max_iter=5)

import matplotlib.pyplot as plt

plt.plot(pso.gbest_y_hist)
plt.savefig('./pso.jpg')
plt.show()