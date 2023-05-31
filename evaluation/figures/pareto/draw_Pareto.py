#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据
end_data = pd.read_csv('./ObjV_wer_red.csv')
all_data = pd.read_csv('./ObjV_wer_all.csv')
asv_socres1 = np.array(end_data)[:, 0]
wer1 = np.array(end_data)[:, 1]
asv_socres2 = np.array(all_data)[:, 0]
wer2 = np.array(all_data)[:, 1]
stoi = []

# 绘制散点图
plt.plot(asv_socres2, wer2, 'o',color='gray')
plt.plot(asv_socres1, wer1, 'o',color='red')

# 添加标签和标题
plt.xlabel('ASV_score', fontsize=16)
plt.ylabel('WER', fontsize=16)

# 显示图形
plt.savefig('wer_asv.png')
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据
end_data = pd.read_csv('./ObjV_stoi_red.csv')
all_data = pd.read_csv('./ObjV_stoi_all.csv')
asv_socres1 = np.array(end_data)[:, 0]
wer1 = np.array(end_data)[:, 1]
asv_socres2 = np.array(all_data)[:, 0]
wer2 = np.array(all_data)[:, 1]
stoi = []

# 绘制散点图
plt.plot(asv_socres2, wer2, 'o',color='gray')
plt.plot(asv_socres1, wer1, 'o',color='red')

# 添加标签和标题
plt.xlabel('ASV_score', fontsize=16)
plt.ylabel('STOI', fontsize=16)

# 显示图形
plt.savefig('stoi_asv.png')
# %%
