#%%
ecapa_eer = [0.7, 15.75, 19.5, 8.99, 10.99]
xvector_eer = [2.5, 19.77, 17.87, 21.42, 19.84]

ecapa_mmr = [0.5, 62, 63, 42.25, 43.5]
xvector_mmr = [2.5, 98.053, 96.298, 90.5, 95.784]

crdnn_wer = [3, 22.55, 24.67, 43.58]
wav2wec_wer = [2, 2.01, 3.55, 9.33]

libri_mean = [0.761, 0.8612, 0.6953]
vox_mean = [0.788, 0.7817, 0.6941]
vctk_mean = [0.796, 0.8259, 0.7526]
aishell_mean = [0.788, 0.7532, 0.6966]

libri_std = [0.0424, 0.0332, 0.0613]
vox_std = [0.0400, 0.0436, 0.0515]
vctk_std = [0.050, 0.039, 0.0463]
aishell_std = [0.0462, 0.0402, 0.0476]

import numpy as np
import matplotlib.pyplot as plt

# 数据准备

x_labels = ['origin', 'Task1', 'Task2', 'McAdams', 'VoiceMask']


# 设置柱子宽度和位置
bar_width = 0.35
x_positions1 = np.arange(5)
x_positions2 = x_positions1 + bar_width

# 绘制柱状图
plt.bar(x_positions1, ecapa_eer, width=bar_width, label='ecapa',color='#FFBE7A')
plt.bar(x_positions2, xvector_eer, width=bar_width,label='xvector',color='#FA7F6F')

# 设置X轴刻度和标签
plt.xticks(x_positions1 + bar_width / 2, x_labels,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('EER(%)',fontsize=12)
# 添加图例
plt.legend(fontsize=12)
plt.savefig('eer.png')
# %%

# 绘制柱状图
plt.bar(x_positions1, ecapa_mmr, width=bar_width, label='ecapa',color='#FFBE7A')
plt.bar(x_positions2, xvector_mmr, width=bar_width,label='xvector',color='#FA7F6F')

# 设置X轴刻度和标签
plt.xticks(x_positions1 + bar_width / 2, x_labels,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('MMR(%)',fontsize=12)
# 添加图例
plt.legend(fontsize=12)
plt.savefig('mmr.png')
# %%

x_labels = ['origin', 'Task2', 'McAdams', 'VoiceMask']

# 设置柱子宽度和位置
bar_width = 0.35
x_positions1 = np.arange(4)
x_positions2 = x_positions1 + bar_width
# 绘制柱状图
plt.bar(x_positions1, wav2wec_wer, width=bar_width, label='wav2wec',color='#FFBE7A')
plt.bar(x_positions2, crdnn_wer, width=bar_width,label='crdnn',color='#FA7F6F')

# 设置X轴刻度和标签
plt.xticks(x_positions1 + bar_width / 2, x_labels,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('WER(%)',fontsize=12)
# 添加图例
plt.legend(fontsize=12)
plt.savefig('wer.png')
# %%
