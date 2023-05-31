#%%
ecapa_eer = [0.7, 16.25, 16.25, 8.99, 10.99]
xvector_eer = [2.5, 22.5, 21.25, 21.42, 19.84]

ecapa_mmr = [0.5, 48.73, 59.07, 42.25, 43.5]
xvector_mmr = [2.5, 95.50, 96.25, 90.5, 95.784]

crdnn_wer = [3, 36.54, 24.67, 43.58]
wav2wec_wer = [2, 9.01, 3.55, 9.33]

libri_stoi = [0.768, 0.8612, 0.6953]

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
