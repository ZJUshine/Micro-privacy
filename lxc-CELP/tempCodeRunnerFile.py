from CELP import CELP
codec = CELP(wave_path=r'/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac',
                save_path=r'61-70968-0000test.flac',anony=False,anonypara=[0.5,0.5])
_,lsfs = codec.run()
print(lsfs[0])