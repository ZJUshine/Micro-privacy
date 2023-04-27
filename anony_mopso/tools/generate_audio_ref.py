'''
FilePath: generate_audio_ref.py
Author: zjushine
Date: 2023-04-25 16:31:09
LastEditors: zjushine
LastEditTime: 2023-04-25 22:47:07
Description: 将librispeech数据集中的音频文件名和对应的文本内容合并到一个文件中，方便后续处理
Copyright (c) 2023 by ${zjushine}, All Rights Reserved. 
'''
import os

# 指定目录路径
txt_dir_path = '/mnt/lxc/librispeech/test-clean/test-clean-audio'

# 递归函数获取所有子文件夹内的txt路径
def get_txt_files(txt_dir_path):
    txt_files = []
    for dirpath, dirnames, filenames in os.walk(txt_dir_path):
        for filename in filenames:
            # 判断文件是否为txt文件
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
        for dirname in dirnames:
            txt_files.extend(get_txt_files(os.path.join(dirpath, dirname)))
    return txt_files

# 获取所有子文件夹内的txt路径
txt_files = get_txt_files(txt_dir_path)
print(txt_dir_path)
# 指定输出文件路径
output_file = 'test-clean-trans.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    # 遍历文件名,构建文件路径
    for txt_file in txt_files:
        for line in open(txt_file):
            if (len(line) > 100):
                continue
            # 找到第一个空格的位置
            index = line.find(' ')
            # 将第一个空格替换为逗号
            line = line[:index] + ',' + line[index + 1:]
            f.writelines(line)
