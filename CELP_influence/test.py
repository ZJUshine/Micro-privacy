import soundfile as sf
import pystoi
import torch
# 导入speechbrain预训练模型
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../pretrained_models/spkrec-ecapa-voxceleb")
from speechbrain.pretrained import EncoderASR
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="../pretrained_models/asr-wav2vec2-librispeech")


orig_data, sr = sf.read(wave_path)
code_data, sr = sf.read(f'/mnt/lxc/Micro-privacy_data/CELP_code/{audio_name}.flac')
score, prediction = verification.verify_batch(torch.tensor(orig_data),torch.tensor(code_data))
stoi_value = pystoi.stoi(orig_data, code_data, sr, extended=False)