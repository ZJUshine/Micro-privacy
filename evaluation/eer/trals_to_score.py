import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
import numpy as np
from tqdm import tqdm
import soundfile as sf
from CELP import CELP
# from pyannote.audio import Inference
import warnings
warnings.filterwarnings("ignore")
import torch, torchaudio
sim_cal = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
torch.set_num_threads(6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------pyannote-----------------------------------------------------
# inference = Inference("pyannote/embedding", window="whole", device=device)
# df = pd.read_csv(r"/home/usslab/Documents2/xinfeng/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# print(df)
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])#; df1.columns=['target', 'wav1', 'wav2']
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     embed1 = inference(row['wav1'])
#     embed2 = inference(row['wav2'])
#     distance = sim_cal(embed1, embed2)
#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)    
# df1.to_csv('df1.txt', sep=' ',index=False)
# df2.to_csv('df2.txt', sep=' ',index=False)

x = [-2.051060100256165905e-02,1.868469258204319416e-02,-1.033536037095581818e-02,7.895556476526205403e-02,\
1.082179264615995389e-02,8.754378824390215974e-02,1.390487721189856418e-01,1.786383073776960095e-01,\
8.572017773985862732e-03,1.089513754007719742e-01]

# # ---------------------------------ECAPA-TDNN-----------------------------------------------------
# from speechbrain.pretrained import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="../../pretrained_models/spkrec-ecapa-voxceleb")
# df = pd.read_csv(r"./df_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1, sr = sf.read(row['wav1'])
#     codec = CELP(wave_path = row['wav2'],save_path=f'./temp_anony.flac',anonypara=x)
#     _,lsfs,modif_lsfs = codec.run()
#     wav2, sr = sf.read(f'./temp_anony.flac')
#     embed1 = classifier.encode_batch(torch.tensor(wav1))
#     embed2 = classifier.encode_batch(torch.tensor(wav2))
#     distance = sim_cal(embed1, embed2)
#     print(distance, row['target'])
#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('df1_libri_tdnn.txt', sep=' ',index=False)
# df2.to_csv('df2_libri_tdnn.txt', sep=' ',index=False)

# # ---------------------------------X-vector-----------------------------------------------------
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="../../pretrained_models/spkrec-xvect-voxceleb")
sim_cal = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
df = pd.read_csv(r"./df_trial.txt", encoding='utf-8', sep=' ')
df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
for i, row in tqdm(df.iterrows()):
    wav1, sr = sf.read(row['wav1'])
    codec = CELP(wave_path = row['wav2'],save_path=f'./temp_anony.flac',anonypara=x)
    _,lsfs,modif_lsfs = codec.run()
    wav2, sr = sf.read(f'./temp_anony.flac')
    embed1 = classifier.encode_batch(torch.tensor(wav1))
    embed2 = classifier.encode_batch(torch.tensor(wav2))
    distance = sim_cal(embed1, embed2)
    print(distance, row['target'])
    df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
    df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
print(df1)
print(df2)
df1.to_csv('df1_libri_xvector.txt', sep=' ',index=False)
df2.to_csv('df2_libri_xvector.txt', sep=' ',index=False)


# # ---------------------------------SB Xvectors-----------------------------------------------------
# from speechbrain.pretrained import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device})
# df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1 = torchaudio.load(row['wav1'])[0]
#     wav2 = torchaudio.load(row['wav2'])[0]
#     wav1 = wav1.to(device)
#     wav2 = wav2.to(device)
#     embed1 = classifier.encode_batch(wav1)
#     embed2 = classifier.encode_batch(wav2)
#     distance = sim_cal(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('librispeech/df1_libri_sbXvec.txt', sep=' ',index=False)
# df2.to_csv('librispeech/df2_libri_sbXvec.txt', sep=' ',index=False)

# # ---------------------------------D-Vectors for Libri-----------------------------------------------------
# wav2mel = torch.jit.load("/home/lxf/Documents/xinfeng/research/ultra_asv_attack/journal/digtal/d-Vec-model/wav2mel.pt")
# dvector = torch.jit.load("/home/lxf/Documents/xinfeng/research/ultra_asv_attack/journal/digtal/d-Vec-model/dvector-step250000.pt").eval()

# df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1 = torchaudio.load(row['wav1'])[0]
#     wav2 = torchaudio.load(row['wav2'])[0]
#     embed1 = dvector.embed_utterance(wav2mel(wav1, 16000))
#     embed2 = dvector.embed_utterance(wav2mel(wav2, 16000))
#     distance = sim_cal(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('librispeech/df1_libri_dVec.txt', sep=' ',index=False)
# df2.to_csv('librispeech/df2_libri_dVec.txt', sep=' ',index=False)

# # ---------------------------------D-Vectors for VoxCeleb-----------------------------------------------------
# wav2mel = torch.jit.load("/home/lxf/Documents/xinfeng/research/ultra_asv_attack/journal/digtal/d-Vec-model/wav2mel.pt")
# dvector = torch.jit.load("/home/lxf/Documents/xinfeng/research/ultra_asv_attack/journal/digtal/d-Vec-model/dvector-step250000.pt").eval()

# df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/voxceleb1/vox1_veri_test2.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1 = torchaudio.load(row['wav1'])[0]
#     wav2 = torchaudio.load(row['wav2'])[0]
#     embed1 = dvector.embed_utterance(wav2mel(wav1, 16000))
#     embed2 = dvector.embed_utterance(wav2mel(wav2, 16000))
#     distance = sim_cal(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('voxceleb1/df1_vox1_dVec.txt', sep=' ',index=False)
# df2.to_csv('voxceleb1/df2_vox1_dVec.txt', sep=' ',index=False)

# # ---------------------------------nemo speakernet for Libri-----------------------------------------------------
# import nemo.collections.asr as nemo_asr
# classifier = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")
# classifier.freeze()

# df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1 = torchaudio.load(row['wav1'])[0].to(device)
#     wav2 = torchaudio.load(row['wav2'])[0].to(device)

#     cur_length = torch.tensor([wav1.shape[1]]).to(device)
#     embed1 = classifier.forward(input_signal=wav1, input_signal_length=cur_length)[1]
#     cur_length = torch.tensor([wav2.shape[1]]).to(device)
#     embed2 = classifier.forward(input_signal=wav2, input_signal_length=cur_length)[1]

#     distance = sim_cal(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('df1_libri_spknet.txt', sep=' ',index=False)
# df2.to_csv('df2_libri_spknet.txt', sep=' ',index=False)

# # ---------------------------------nemo speakernet for VoxCeleb-----------------------------------------------------
# import nemo.collections.asr as nemo_asr
# classifier = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")
# classifier.freeze()

# df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/voxceleb1/vox1_veri_test2.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1 = torchaudio.load(row['wav1'])[0].to(device)
#     wav2 = torchaudio.load(row['wav2'])[0].to(device)

#     cur_length = torch.tensor([wav1.shape[1]]).to(device)
#     embed1 = classifier.forward(input_signal=wav1, input_signal_length=cur_length)[1]
#     cur_length = torch.tensor([wav2.shape[1]]).to(device)
#     embed2 = classifier.forward(input_signal=wav2, input_signal_length=cur_length)[1]

#     distance = sim_cal(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('voxceleb1/df1_vox1_spknet.txt', sep=' ',index=False)
# df2.to_csv('voxceleb1/df2_vox1_spknet.txt', sep=' ',index=False)

# # ---------------------------------nemo titanet-----------------------------------------------------
# import nemo.collections.asr as nemo_asr
# classifier = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
# classifier.freeze()

# df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     wav1 = torchaudio.load(row['wav1'])[0].to(device)
#     wav2 = torchaudio.load(row['wav2'])[0].to(device)

#     cur_length = torch.tensor([wav1.shape[1]]).to(device)
#     embed1 = classifier.forward(input_signal=wav1, input_signal_length=cur_length)[1]
#     cur_length = torch.tensor([wav2.shape[1]]).to(device)
#     embed2 = classifier.forward(input_signal=wav2, input_signal_length=cur_length)[1]

#     distance = sim_cal(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('librispeech/df1_libri_titanet.txt', sep=' ',index=False)
# df2.to_csv('librispeech/df2_libri_titanet.txt', sep=' ',index=False)

# # ---------------------------------DeepSpeaker-----------------------------------------------------
# import numpy as np
# import sys
# sys.path.append('/root/research/deepspeaker/')
# from deep_speaker.audio import read_mfcc
# from deep_speaker.batcher import sample_from_mfcc
# from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
# from deep_speaker.conv_models import DeepSpeakerModel
# from deep_speaker.test import batch_cosine_similarity
# model = DeepSpeakerModel()
# # Load the checkpoint.
# model.m.load_weights('/root/research/deepspeaker/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

# df = pd.read_csv(r"/root/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):
#     mfcc_001 = sample_from_mfcc(read_mfcc(row['wav1'], SAMPLE_RATE), 400)
#     mfcc_002 = sample_from_mfcc(read_mfcc(row['wav2'], SAMPLE_RATE), 400)
    
#     embed1 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
#     print(embed1.shape)
#     embed2 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

#     distance = batch_cosine_similarity(embed1, embed2)

#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': distance.item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('librispeech/df1_libri_deepspk2.txt', sep=' ',index=False)
# df2.to_csv('librispeech/df2_libri_deepspk2.txt', sep=' ',index=False)



# ---------------------------------VGGspeaker-----------------------------------------------------
# import numpy as np
# import sys
# sys.path.append('/root/research/VGG-Speaker-Recognition/src')
# import utils, model

# import argparse
# parser = argparse.ArgumentParser()
# # set up training configuration.
# parser.add_argument('--gpu', default='', type=str)
# parser.add_argument('--resume', default='/root/research/VGG-Speaker-Recognition/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5', type=str)
# parser.add_argument('--batch_size', default=16, type=int)
# parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
# # set up network configuration.
# parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
# parser.add_argument('--ghost_cluster', default=2, type=int)
# parser.add_argument('--vlad_cluster', default=8, type=int)
# parser.add_argument('--bottleneck_dim', default=512, type=int)
# parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# # set up learning rate, training loss and optimizer.
# parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
# parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

# global args
# args = parser.parse_args()

# # ==================================
# #       Get Model
# # ==================================
# # construct the data generator.
# params = {'dim': (257, None, 1),
#             'nfft': 512,
#             'spec_len': 250,
#             'win_length': 400,
#             'hop_length': 160,
#             'n_classes': 5994,
#             'sampling_rate': 16000,
#             'normalize': True,
#             }

# network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
#                                             num_class=params['n_classes'],
#                                             mode='eval', args=args)
# network_eval.load_weights(os.path.join(args.resume), by_name=True)

# df = pd.read_csv(r"librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
# df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
# for i, row in tqdm(df.iterrows()):


#     specs_1 = utils.load_data(row['wav1'], win_length=params['win_length'], sr=params['sampling_rate'],
#                              hop_length=params['hop_length'], n_fft=params['nfft'],
#                              spec_len=params['spec_len'], mode='eval')
#     specs_2 = utils.load_data(row['wav2'], win_length=params['win_length'], sr=params['sampling_rate'],
#                              hop_length=params['hop_length'], n_fft=params['nfft'],
#                              spec_len=params['spec_len'], mode='eval')
#     specs_1 = np.expand_dims(np.expand_dims(specs_1, 0), -1)
#     specs_2 = np.expand_dims(np.expand_dims(specs_2, 0), -1)

#     embed1 = network_eval.predict(specs_1)
#     embed2 = network_eval.predict(specs_2)
#     score = np.sum(embed1 * embed2)
#     df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     df2 = df2.append({'target': score, 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
# print(df1)
# print(df2)
# df1.to_csv('librispeech/df1_libri_VGGspk.txt', sep=' ',index=False)
# df2.to_csv('librispeech/df2_libri_VGGspk.txt', sep=' ',index=False)

# # --------------------------------- WavLM Model for Libri-----------------------------------------------------
# # 模型比较大，跑起来也挺慢的
# with torch.no_grad():
#     classifier = torch.load('/home/lxf/Documents/xinfeng/research/ultra_asv_attack/playground/sidekit/best_model_wavlm_cuda_JIT.pt')
#     classifier.eval()

#     df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/librispeech/lib_test_trial.txt", encoding='utf-8', sep=' ')
#     df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
#     df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
#     for i, row in tqdm(df.iterrows()):
#         wav1 = torchaudio.load(row['wav1'])[0].to(device)
#         wav2 = torchaudio.load(row['wav2'])[0].to(device)

#         embed1 = classifier(wav1)[1]
#         embed2 = classifier(wav2)[1]

#         distance = sim_cal(embed1, embed2)

#         df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#         df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     print(df1)
#     print(df2)
#     df1.to_csv('librispeech/df1_libri_WavLM.txt', sep=' ',index=False)
#     df2.to_csv('librispeech/df2_libri_WavLM.txt', sep=' ',index=False)

# # --------------------------------- WavLM Model for VoxCeleb (注意这个代码版本和31服务器的base环境配合，和nemo不配合) -----------------------------------------------------
# # 模型比较大，跑起来也挺慢的
# with torch.no_grad():
#     classifier = torch.load('/home/lxf/Documents/xinfeng/research/ultra_asv_attack/playground/sidekit/best_model_wavlm_cuda_JIT.pt')
#     classifier.eval()

#     df = pd.read_csv(r"/home/lxf/Documents/xinfeng/research/ultra_asv_attack/eer_calculate/voxceleb1/vox1_veri_test2.txt", encoding='utf-8', sep=' ')
#     df1 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
#     df2 = pd.DataFrame(columns=['target', 'wav1', 'wav2'])
#     for i, row in tqdm(df.iterrows()):
#         wav1 = torchaudio.load(row['wav1'])[0].to(device)
#         wav2 = torchaudio.load(row['wav2'])[0].to(device)

#         embed1 = classifier(wav1)[1]
#         embed2 = classifier(wav2)[1]

#         distance = sim_cal(embed1, embed2)

#         df1 = df1.append({'target': row['target'], 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#         df2 = df2.append({'target': distance.cpu().detach().numpy().item(), 'wav1': row['wav1'], 'wav2': row['wav2']}, ignore_index=True)
#     print(df1)
#     print(df2)
#     df1.to_csv('voxceleb1/df1_vox1_WavLM.txt', sep=' ',index=False)
#     df2.to_csv('voxceleb1/df2_vox1_WavLM.txt', sep=' ',index=False)