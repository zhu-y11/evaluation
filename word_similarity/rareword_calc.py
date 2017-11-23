# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Rareword Evaluation
@Author Yi Zhu
Upated 20.11.2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch

from word_similarity.ws_utils import findIntersec, calcSpearmanr

def rarewordCalc(evaldata_file_path, emb_vocab, emb_vectors, lower_case):
  word_pairs, sims = readRareData(evaldata_file_path, lower_case)
  print('finding intersecting coverage ...')
  inter_vocab, inter_vectors, inter_word_pairs, inter_sims = findIntersec(emb_vocab, emb_vectors, word_pairs, sims)
  return (len(word_pairs), calcSpearmanr(emb_vocab, emb_vectors, word_pairs, sims)),\
         (len(inter_word_pairs), calcSpearmanr(inter_vocab, inter_vectors, inter_word_pairs, inter_sims))   
  
  
def readRareData(data_file_path, lower_case):
  with open(data_file_path, 'r') as f:
    lines = f.read().strip().split('\n')
    lines = [line.strip().split() for line in lines]
    word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]
    sims = torch.Tensor(list(map(float, [line[-1] for line in lines])))
    return word_pairs, sims
