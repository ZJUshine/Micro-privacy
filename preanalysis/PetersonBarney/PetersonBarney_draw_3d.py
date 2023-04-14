'''
FilePath: PetersonBarney_draw_3d.py
Author: zjushine
Date: 2023-04-14 17:48:23
LastEditors: zjushine
LastEditTime: 2023-04-14 21:25:33
Description: 数据集：PetersonBarney的3D散点图（F1,F2,F3）
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
import pandas as pd
import matplotlib.pyplot as plt
# 数据集路径
path = "./verified_pb.txt"
df = pd.read_csv(path, header=None,sep="\t")
#解决不能显示中文的问题
plt.rcParams['axes.unicode_minus']=False

df.columns = ['0','1','2','3','4','5','6','7']

#画多组数据的散点图
IYx = [];IYy = [];IYz = []
IHx = [];IHy = [];IHz = []
EHx = [];EHy = [];EHz = []
AEx = [];AEy = [];AEz = []
AHx = [];AHy = [];AHz = []
AAx = [];AAy = [];AAz = []
AOx = [];AOy = [];AOz = []
UHx = [];UHy = [];UHz = []
UWx = [];UWy = [];UWz = []
ERx = [];ERy = [];ERz = []

for i in range(len(df)):
    if (df["3"][i] == "IY" or df["3"][i] == "*IY"):
        IYx.append(df["5"][i])
        IYy.append(df["6"][i])
        IYz.append(df["7"][i])
    elif (df["3"][i] == "IH" or df["3"][i] == "*IH"):
        IHx.append(df["5"][i])
        IHy.append(df["6"][i])
        IHz.append(df["7"][i])
    elif (df["3"][i] == "EH" or df["3"][i] == "*EH"):
        EHx.append(df["5"][i])
        EHy.append(df["6"][i])
        EHz.append(df["7"][i])
    elif (df["3"][i] == "AE" or df["3"][i] == "*AE"):
        AEx.append(df["5"][i])
        AEy.append(df["6"][i])
        AEz.append(df["7"][i])
    elif (df["3"][i] == "AH" or df["3"][i] == "*AH"):
        AHx.append(df["5"][i])
        AHy.append(df["6"][i])
        AHz.append(df["7"][i])     
    elif (df["3"][i] == "AA" or df["3"][i] == "*AA"):
        AAx.append(df["5"][i])
        AAy.append(df["6"][i])
        AAz.append(df["7"][i])
    elif (df["3"][i] == "AO" or df["3"][i] == "*AO"):
        AOx.append(df["5"][i])
        AOy.append(df["6"][i])
        AOz.append(df["7"][i])
    elif (df["3"][i] == "UH" or df["3"][i] == "*UH"):
        UHx.append(df["5"][i])
        UHy.append(df["6"][i])
        UHz.append(df["7"][i])
    elif (df["3"][i] == "UW" or df["3"][i] == "*UW"):
        UWx.append(df["5"][i])
        UWy.append(df["6"][i])
        UWz.append(df["7"][i])
    elif (df["3"][i] == "ER" or df["3"][i] == "*ER"):
        ERx.append(df["5"][i])
        ERy.append(df["6"][i])
        ERz.append(df["7"][i])

s_size=10
plt.figure(figsize=(6,6))

ax = plt.axes(projection="3d")
ax.scatter3D(IYx,IYy,IYz,color=(0/255,168/255,225/255),alpha=0.5,label="IY",s = s_size)
ax.scatter3D(IHx,IHy,IHz,color=(153/255,204/255,0),alpha=0.5,label="IH",s = s_size)
ax.scatter3D(EHx,EHy,EHz,color=(227/255,0,57/255),alpha=0.5,label="EH",s = s_size)
ax.scatter3D(AEx,AEy,AEz,color=(252/255,211/255,0),alpha=0.5,label="AE",s = s_size)
ax.scatter3D(AHx,AHy,AHz,color=(128/255,0,128/255),alpha=0.5,label="AH",s = s_size)
ax.scatter3D(AAx,AAy,AAz,color=(0,153/255,78/255),alpha=0.5,label="AA",s = s_size)
ax.scatter3D(AOx,AOy,AOz,color=(255/255,102/255,0),alpha=0.5,label="AO",s = s_size)
ax.scatter3D(UHx,UHy,UHz,color=(128/255,128/255,0),alpha=0.5,label="UH",s = s_size)
ax.scatter3D(UWx,UWy,UWz,color=(219/255,0,194/255),alpha=0.5,label="UW",s = s_size)
ax.scatter3D(ERx,ERy,ERz,color=(0,0,255/255),alpha=0.5,label="ER",s = s_size)


ax.set_xlabel('F1(Hz)',fontsize = 10)
ax.set_ylabel('F2(Hz)',fontsize = 10)
ax.set_zlabel('F3(Hz)',fontsize = 10)

#显示图例 
plt.legend(loc="upper left",fontsize=10)
plt.grid()
plt.savefig(f"./fig/PetersonBarney_3d.png")
plt.clf()

