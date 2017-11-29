# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
CONLLU Loader
@Author Yi Zhu
Upated 28/11/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

def loadCONLLUFromDir(dir_path):
  for conllu_file in os.listdir(dir_path):
    if not conllu_file.endswith('.conllu'):
      continue
    if 'train' in conllu_file:
      train_data = loadCONLLUFromFile(os.path.join(dir_path, conllu_file))
    if 'dev' in conllu_file:
      dev_data = loadCONLLUFromFile(os.path.join(dir_path, conllu_file))
    if 'test' in conllu_file:
      test_data = loadCONLLUFromFile(os.path.join(dir_path, conllu_file))
  return train_data, dev_data, test_data


def loadCONLLUFromFile(file_path):
  with open(file_path, 'r') as f:
    sents = f.read().strip().split('\n\n')
    for i, sent in enumerate(sents[:]):
      conllu_sent = sent.strip().split('\n')
      conllu_sent = [line.strip().split('\t') for line in conllu_sent if not line.startswith('#')]
      sents[i] = conllu_sent
  return sents
