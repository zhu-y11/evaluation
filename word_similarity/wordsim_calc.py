# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Wordsim353 Evaluation
@Author Yi Zhu
Upated 21/11/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch
import torch.nn.functional as F

import metrics
from word_similarity.ws_utils import findIntersec, calcSpearmanr
from lang_map import lang_map


def wordSimCalc(evaldata_file_path, emb_vocab, emb_vectors, lower_case):
  word_pairs, sims = readData(evaldata_file_path, lower_case)
  print('finding intersecting coverage ...')
  inter_vocab, inter_vectors, inter_word_pairs, inter_sims = findIntersec(emb_vocab, emb_vectors, word_pairs, sims)
  return (len(word_pairs), calcSpearmanr(emb_vocab, emb_vectors, word_pairs, sims)),\
         (len(inter_word_pairs), calcSpearmanr(inter_vocab, inter_vectors, inter_word_pairs, inter_sims))   
  

def wordSimCalcRelSim(evaldata_file_path, emb_vocab, emb_vectors, lower_case):
  word_pairs, sims = readNoHeadData(evaldata_file_path, lower_case)
  print('finding intersecting coverage ...')
  inter_vocab, inter_vectors, inter_word_pairs, inter_sims = findIntersec(emb_vocab, emb_vectors, word_pairs, sims)
  return (len(word_pairs), calcSpearmanr(emb_vocab, emb_vectors, word_pairs, sims)),\
         (len(inter_word_pairs), calcSpearmanr(inter_vocab, inter_vectors, inter_word_pairs, inter_sims))   


def multWordSimCalc(evaldata_dir_path, embs_map, lower_case):
  data_map = readMultData(evaldata_dir_path, lower_case)
  print('finding intersecting coverage ...')
  for lang in data_map:
    if lang not in embs_map:
      continue
    n_pair = len(data_map[lang][0])
    spr = calcSpearmanr(embs_map[lang][0], embs_map[lang][1], data_map[lang][0], data_map[lang][1])
    inter_vocab, inter_vectors, inter_word_pairs, inter_sims = findIntersec(embs_map[lang][0],
                                                                            embs_map[lang][1],
                                                                            data_map[lang][0],
                                                                            data_map[lang][1])
    n_int_pair = len(inter_word_pairs)
    int_spr = calcSpearmanr(inter_vocab, inter_vectors, inter_word_pairs, inter_sims)

    print('{} {} {:.5f}'.format(lang, n_pair, spr))
    print('{} {} {:.5f}\n'.format(lang, n_int_pair, int_spr))


def multWordSimCalcRelSim(evaldata_dir_path, embs_map, lower_case):
  data_map = readMultDataRelSim(evaldata_dir_path, lower_case)
  print('finding intersecting coverage ...')
  for lang in data_map:
    if lang not in embs_map:
      continue
    n_pair = len(data_map[lang][0])
    spr = calcSpearmanr(embs_map[lang][0], embs_map[lang][1], data_map[lang][0], data_map[lang][1])
    inter_vocab, inter_vectors, inter_word_pairs, inter_sims = findIntersec(embs_map[lang][0],
                                                                            embs_map[lang][1],
                                                                            data_map[lang][0],
                                                                            data_map[lang][1])
    n_int_pair = len(inter_word_pairs)
    int_spr = calcSpearmanr(inter_vocab, inter_vectors, inter_word_pairs, inter_sims)

    print('{} {} {:.5f}'.format(lang, n_pair, spr))
    print('{} {} {:.5f}\n'.format(lang, n_int_pair, int_spr))


def readData(data_file_path, lower_case):
  with open(data_file_path, 'r') as f:
    lines = f.read().strip().split('\n')
    lines = lines[1:]
    lines = [line.strip().split() for line in lines]
    word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]
    sims = torch.Tensor(list(map(float, [line[-1] for line in lines])))
    return word_pairs, sims


def readNoHeadData(data_file_path, lower_case):
  with open(data_file_path, 'r') as f:
    lines = f.read().strip().split('\n')
    lines = [line.strip().split() for line in lines]
    word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]
    sims = torch.Tensor(list(map(float, [line[-1] for line in lines])))
    return word_pairs, sims


def readMultData(dir_path, lower_case):
  data_map = {}
  for data_file in os.listdir(dir_path):
    if not data_file.endswith('.txt'):
      continue
    print(data_file)
    try:
      lang = lang_map[data_file[data_file.find('_') + 1: data_file.rfind('.')].lower()]
    except:
      continue
    print(lang)
    with open(os.path.join(dir_path, data_file), 'r') as f:
      lines = f.read().strip().split('\n')
      lines = lines[1:]
      lines = [line.strip().split(',') for line in lines]
      word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]
      sims = torch.Tensor(list(map(float, [line[-1] for line in lines])))
      data_map[lang] = word_pairs, sims
  return data_map


def readMultDataRelSim(dir_path, lower_case):
  data_map = {}
  for data_file in os.listdir(dir_path):
    if not data_file.endswith('.txt'):
      continue
    try:
      lang = lang_map[data_file[data_file.find('-') + 1: data_file.rfind('-')].lower()]
    except:
      continue
    with open(os.path.join(dir_path, data_file), 'r') as f:
      lines = f.read().strip().split('\n')
      lines = lines[1:]
      lines = [line.strip().split(',') for line in lines]
      word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]
      sims = torch.Tensor(list(map(float, [line[-1] for line in lines])))
      data_map[lang] = word_pairs, sims
  return data_map
