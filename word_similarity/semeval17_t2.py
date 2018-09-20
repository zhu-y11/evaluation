# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Semeval 17 Task2 Evaluation
@Author Yi Zhu
Upated 21/11/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os
import numpy as np

import torch

from word_similarity.ws_utils import findIntersec, calcSpearmanr, findBiIntersec, calcBiSpearmanr


def semEval17T2Calc(evaldata_dir_path, embs_map, lower_case):
  data_map = readData(evaldata_dir_path, lower_case)
  print('finding intersecting coverage ...')
  for dt in data_map:
    for lang1 in data_map[dt]:
      if lang1 not in embs_map:
        continue
      for lang2 in data_map[dt][lang1]:
        if lang2 not in embs_map:
          continue
        if lang1 == lang2:
          n_pair = len(data_map[dt][lang1][lang2][0])
          spr = calcSpearmanr(embs_map[lang1][0], embs_map[lang1][1], data_map[dt][lang1][lang2][0], data_map[dt][lang1][lang2][1])
          inter_vocab, inter_vectors, inter_word_pairs, inter_sims = findIntersec(embs_map[lang1][0],
                                                                                  embs_map[lang1][1],
                                                                                  data_map[dt][lang1][lang2][0],
                                                                                  data_map[dt][lang1][lang2][1])
          n_int_pair = len(inter_word_pairs)
          int_spr = calcSpearmanr(inter_vocab, inter_vectors, inter_word_pairs, inter_sims)
        else:
          n_pair = len(data_map[dt][lang1][lang2][0])
          spr = calcBiSpearmanr(embs_map[lang1][0], embs_map[lang1][1], 
                                embs_map[lang2][0], embs_map[lang2][1],
                                data_map[dt][lang1][lang2][0], data_map[dt][lang1][lang2][1])
          lang1_inter_vocab, lang1_inter_vectors,\
          lang2_inter_vocab, lang2_inter_vectors,\
          inter_word_pairs, inter_sims = findBiIntersec(embs_map[lang1][0],
                                                        embs_map[lang1][1],
                                                        embs_map[lang2][0],
                                                        embs_map[lang2][1],
                                                        data_map[dt][lang1][lang2][0],
                                                        data_map[dt][lang1][lang2][1])
          n_int_pair = len(inter_word_pairs)
          int_spr = calcBiSpearmanr(lang1_inter_vocab, lang1_inter_vectors, 
                                    lang2_inter_vocab, lang2_inter_vectors,
                                    inter_word_pairs, inter_sims)
        print('{} {}-{} {} {:.5f}'.format(dt, lang1, lang2, n_pair, spr))
        print('{} {}-{} {} {:.5f}\n'.format(dt, lang1, lang2, n_int_pair, int_spr)) 
  

def readData(data_dir_path, lower_case):
  data_map = {}
  data_types = ['trial', 'test']
  sub_tasks = ['subtask1-monolingual', 'subtask2-crosslingual']
  for dt in data_types:
    data_map[dt] = {}
    for st in sub_tasks:
      st_data_dir_path = os.path.join(data_dir_path, dt, st, 'data')
      st_key_dir_path = os.path.join(data_dir_path, dt, st, 'keys')
      for data_file in os.listdir(st_data_dir_path):
        langs = data_file[:data_file.find('.')].split('-')
        langs = langs + langs if len(langs) == 1 else langs
        key_file = data_file.replace('data', 'gold')

        with open(os.path.join(st_data_dir_path, data_file), 'r') as f:
          lines = f.read().strip().split('\n')
          lines = [line.strip().split() for line in lines]
          word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]

        with open(os.path.join(st_key_dir_path, key_file), 'r') as f:
          lines = f.read().strip().split('\n')
          sims = torch.Tensor(list(map(float, lines)))
        
        assert(sims.size()[0] == len(word_pairs))
        if langs[0] not in data_map[dt]:
          data_map[dt][langs[0]] = {langs[1]: (word_pairs, sims)}
        else:
          data_map[dt][langs[0]][langs[1]] = word_pairs, sims
  return data_map


def transform(embs_map):
  """
  Apply the given transformation to the vector space

  Right-multiplies given transform with embeddings E:
      E = E * transform
  """
  for lang in embs_map:
    transform = os.path.join('alignment_matrices', '{}.txt'.format(lang))
    print('transforming {} ...'.format(lang))
    transmat = torch.Tensor(np.loadtxt(transform))
    embs_map[lang][1] = embs_map[lang][1] @ transmat
