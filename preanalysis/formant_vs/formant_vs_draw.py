'''
FilePath: formant_vs_draw.py
Author: zjushine
Date: 2023-04-14 17:48:23
LastEditors: zjushine
LastEditTime: 2023-04-19 20:58:07
Description: 原始、McAdams、VoiceMask画出频谱观察共振峰
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 创建一个包含两个子图的图形
fig, ax = plt.subplots(2, 3, figsize=(16, 6))

# 读取音频文件
orig_data, sr = sf.read('/mnt/lxc/librispeech/test-clean/test-clean-audio/8230/279154/8230-279154-0001.flac')
mcadams_data, sr = sf.read('/mnt/lxc/Micro-privacy_data/mcadam/8230/279154/8230-279154-0001.wav')
voicemask_data, sr = sf.read('/mnt/lxc/Micro-privacy_data/voicemask/8230/279154/8230-279154-0001.wav')
frange = [37760, 38240]

# 截取音频片段
orig_data = orig_data[frange[0]:frange[1]]
mcadams_data = mcadams_data[frange[0]:frange[1]]
voicemask_data = voicemask_data[frange[0]:frange[1]]

time_axis =np.linspace(0, orig_data.size,orig_data.size)

# 画出原始音频
ax[0, 0].plot(time_axis, orig_data)
ax[0, 1].plot(time_axis, mcadams_data)
ax[0, 2].plot(time_axis, voicemask_data)


# 定义窗口函数
window_size = len(orig_data)
window = np.hanning(window_size)
orig_data *= window
mcadams_data *= window
voicemask_data *= window

# 计算FFT
fft_spectrum_orig = np.fft.fft(orig_data)
fft_spectrum_mcadams = np.fft.fft(mcadams_data)
fft_spectrum_voicemask = np.fft.fft(voicemask_data)

# 计算频率轴
freqs = np.fft.fftfreq(len(orig_data), 1/sr)

ax[1, 0].plot(freqs[:len(freqs)//4], np.abs(fft_spectrum_orig[:len(fft_spectrum_orig)//4]))
ax[1, 1].plot(freqs[:len(freqs)//4], np.abs(fft_spectrum_mcadams[:len(fft_spectrum_mcadams)//4]))
ax[1, 2].plot(freqs[:len(freqs)//4], np.abs(fft_spectrum_voicemask[:len(fft_spectrum_voicemask)//4]))



# 添加标题和轴标签
ax[0, 0].set_title('orig')
ax[0, 1].set_title('mcadams')
ax[0, 2].set_title('voicemask')
ax[1, 0].set_title('fft_spectrum_orig')
ax[1, 1].set_title('fft_spectrum_mcadams')
ax[1, 2].set_title('fft_spectrum_voicemask')

# 调整子图之间的间距
fig.tight_layout()

# 显示图形
#plt.show()
plt.savefig(f"./fig/formant_vs.png")
# %%
