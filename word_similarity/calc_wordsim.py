# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  04/24/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

from word_similarity.ws_utils import findIntersec, calcSpearmanr
import metrics

import logging
logger = logging.getLogger(__name__)

import pdb


def eval_word_similarity(lang, test_data, vocab, emb, lower_case):
  logger.info('Loading test data from {}, language: {}'.format(test_data, lang))
  if not test_data.endswith('.txt'):
    # it is a dir
    test_data_lang_dir = os.path.join(test_data, lang)
    for file_name in sorted(os.listdir(test_data_lang_dir)):
      if not file_name.endswith('.txt'):
        continue
      eval_file_word_similarity(test_data_lang_dir, file_name, vocab, emb, lower_case)
  else:
    # just a file
    eval_file_word_similarity('./', test_data, vocab, emb, lower_case)


def eval_file_word_similarity(test_data_lang_dir, file_name, vocab, emb, lower_case):
  print('Test File: {}'.format(file_name))
  test_file = os.path.join(test_data_lang_dir, file_name) 
  word_pairs, sims = readData(test_file, lower_case)
  logging.debug('finding intersecting coverage...')
  inter_vocab, inter_emb, inter_word_pairs, inter_sims = findIntersec(vocab, emb, word_pairs, sims)
  r = (len(word_pairs), calcSpearmanr(vocab, emb, word_pairs, sims))
  inter_r = (len(inter_word_pairs), calcSpearmanr(inter_vocab, inter_emb, inter_word_pairs, inter_sims))   
  print('{} {:.5f}'.format(r[0], r[1]))
  print('{} {:.5f}'.format(inter_r[0], inter_r[1]))


def readData(data_file_path, lower_case):
  with open(data_file_path, 'r') as f:
    lines = f.read().strip().split('\n')
    lines = [line.strip().split() for line in lines]
    word_pairs = [[line[0].lower(), line[1].lower()] if lower_case else [line[0], line[1]] for line in lines]
    sims = torch.Tensor(list(map(float, [line[-1] for line in lines])))
    return word_pairs, sims
