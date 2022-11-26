#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import models.group1 as models
import numpy as np
from scipy.stats import weightedtau
import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, LogME_optimal, EMMS, TransRate


def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

finetune_acc =  {
                'flickr8k': {'vit_bert': 0.18513, 'vit_roberta': 0.20527, 'vit_bart': 0.21897, 'swinvit_bert': 0.22913, 'swinvit_roberta': 0.23986, 'swinvit_bart': 0.2468, 'swin2vit_bert': 0.25686, 'swin2vit_roberta': 0.23402, 'swin2vit_bart': 0.26235},
                'flickr30k': {'vit_bert':0.26648, 'vit_roberta': 0.23701, 'vit_bart': 0.25134, 'swinvit_bert': 0.26614, 'swinvit_roberta': 0.28838, 'swinvit_bart': 0.28032, 'swin2vit_bert': 0.32331, 'swin2vit_roberta': 0.28814, 'swin2vit_bart': 0.30352}, 
                'RSICD': {'vit_bert': 0.30389, 'vit_roberta': 0.28921, 'vit_bart': 0.30347, 'swinvit_bert': 0.32539,'swinvit_roberta': 0.3207, 'swinvit_bart': 0.31989, 'swin2vit_bert': 0.34449, 'swin2vit_roberta': 0.35218, 'swin2vit_bart': 0.33715},  
                'flickr10kH': {'vit_bert': 0.04312, 'vit_roberta': 0.04882, 'vit_bart': 0.04753, 'swinvit_bert': 0.05245, 'swinvit_roberta': 0.06115, 'swinvit_bart': 0.05099, 'swin2vit_bert': 0.04863, 'swin2vit_roberta': 0.05799, 'swin2vit_bart': 0.069},  
                'flickr10kR': {'vit_bert': 0.04184, 'vit_roberta': 0.0448, 'vit_bart': 0.04526, 'swinvit_bert': 0.04665, 'swinvit_roberta': 0.04492, 'swinvit_bart': 0.04946, 'swin2vit_bert': 0.04489, 'swin2vit_roberta': 0.06134, 'swin2vit_bart': 0.04956},  
                }

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2