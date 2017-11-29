# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Extrinsic Evaluation for word embeddings
@Author Yi Zhu
Upated 29/11/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

import config
import emb_loader
import CONLLU_loader
from pos_tagging_bilstm import evalEmbed 
from lang_map import ud_map


def main(args):
  print('loading word embeddings {} ...'.format(args.embedding_model[0]))
  embs_map = emb_loader.loadEmbeds(args.embedding_model,
                               args.embedding_dir_path,
                               args.embedding_file_name,
                               args.lower_case)
  assert(len(embs_map['en'][0]) == embs_map['en'][1].size()[0])

  loadDataAndEval(args, embs_map)


def loadDataAndEval(args, embs_map):
  #print('loading {} dataset from\n{} ...'.format(args.evaldata_name, args.evaldata_path))
  if args.task == 'pos_tagging':
    lang = ud_map[args.evaldata_name]
    assert(lang in embs_map)
    print('POS tagging in {} ...'.format(lang))
    train_data, dev_data, test_data = CONLLU_loader.loadCONLLUFromDir(args.evaldata_path)
    train_data, dev_data, test_data = train_data[:500], dev_data[:30], test_data[:30]
    evalEmbed(embs_map[lang], train_data, dev_data, test_data) 



if __name__ == '__main__':
  args = config.parse_args()
  main(args)
