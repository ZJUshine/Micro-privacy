#!/usr/bin/python
#-*- coding: utf-8 -*-
# USAGE: python compute_EER.py --ground_truth data/trials.txt --prediction data/scores.txt 

import os
import numpy 
import argparse
import pdb
import random
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

# ==================== === ====================
def GetArgs():
    parser = argparse.ArgumentParser(description = "VoxSRC")
    parser.add_argument('--ground_truth', type=str, default='./df1_libri_xvector.txt', help="Input trials file, with columns of the form "
        "<t0/1> <utt1> <utt2>")
    parser.add_argument('--prediction', type=str, default='./df2_libri_xvector.txt', help="Input scores file, with columns of the form "
        "<score> <utt1> <utt2>")
    parser.add_argument('--positive', type=int, default=1, help='1 if higher is positive; 0 is lower is positive')

    # opt = parser.parse_args()
    args = parser.parse_args()
    return args

# ==================== === ====================

def calculate_eer(y, y_score, pos):
# y denotes groundtruth scores,
# y_score denotes the prediction scores.

	fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
	eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	thresh = interp1d(fpr, thresholds)(eer)

	return eer, thresh

# ==================== === ====================

def read_score(filename):
    scores_file = open(filename, 'r').readlines()
    scores = [] 
	# you may also want to remove whitespace characters like `\n` at the end of each line
    for line in scores_file:
        if "target" in line: continue
        try:
            if 'df2_libri_sbXvec' in filename:
                score, utt1, utt2 = line.rstrip().split()
                _score = (float(score) - 0.858) / (1 - 0.7576)
                print(type(_score))
                scores.append(_score)
            else:
                score, utt1, utt2 = line.rstrip().split()
                scores.append(float(score))                
        except:
            utt1, utt2 = line.rstrip().split()
            scores.append(random.uniform(-1, 1))
    return scores

# ==================== === ====================
def main():
    args = GetArgs()
    y = read_score(args.ground_truth)
    y_score = read_score(args.prediction)
    print(y_score)
    eer, thresh = calculate_eer(y, y_score, args.positive)

    print('EER : %.3f%%, threshold: %.3f'%(eer*100, thresh))

if __name__ == "__main__":
  main()
