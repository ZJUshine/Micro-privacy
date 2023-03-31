from CELP import CELP
import sys
sys.path.append('/home/lxc/zero/speechbrain/pretrained_models')

codec = CELP(wave_path=r'/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac',
                save_path=r'61-70968-0000test.flac')
_,lsfs = codec.run()
print(lsfs.shape)

from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("61-70968-0000.flac", "61-70968-0000test.flac")
print(score, prediction)