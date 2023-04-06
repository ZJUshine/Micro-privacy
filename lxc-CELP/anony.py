from CELP import CELP
from sko.PSO import PSO
import soundfile as sf
import fastwer
import pystoi
import sys
sys.path.append('/home/lxc/zero/speechbrain/pretrained_models')
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")


def demo_func(x):
    origin, sr = sf.read('/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
    deg, sr = sf.read('61-70968-0000test.flac')
    codec = CELP(wave_path=r'/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac',
                    save_path=r'61-70968-0000test.flac',anonypara=x,anony=True)
    ref = ["HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT"]
    _,lsfs = codec.run()
    print(lsfs)
    score, prediction = verification.verify_files("/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac", "61-70968-0000test.flac")
    print(x,score)
    stoi_value = pystoi.stoi(origin, deg, sr, extended=False)
    print(stoi_value)
    asr_result = asr_model.transcribe_file("61-70968-0000test.flac")
    print(asr_result)
    wer = fastwer.score([asr_result], ref)
    print(wer)
    return score+wer

pso = PSO(func=demo_func, n_dim=2, pop=5, max_iter=5, lb=[0.9, -0.1], ub=[1.1, 0.1], w=0.8, c1=2, c2=2)
pso.run(max_iter=5)

import matplotlib.pyplot as plt

plt.plot(pso.gbest_y_hist)
plt.savefig('./pso.jpg')
plt.show()