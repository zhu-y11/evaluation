# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Extrinsic Evaluation for word embeddings
@Author Yi Zhu
Upated 08/12/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

import config, emb_loader, CONLLU_loader, semeval16t5_loader
import pos_tagging_bilstm, sentiment_analysis
from lang_map import ud_map


def main(args):
  print('loading word embeddings {} ...'.format(args.embedding_model[0]))
  embs_map = {}
  embs_map = emb_loader.loadEmbeds(args.embedding_model,
                               args.embedding_dir_path,
                               args.embedding_file_name,
                               args.lower_case)
  loadDataAndEval(args, embs_map)


def loadDataAndEval(args, embs_map):
  cuda = args.cuda if torch.cuda.is_available() else False
  #print('loading {} dataset from\n{} ...'.format(args.evaldata_name, args.evaldata_path))
  if args.task == 'pos_tagging':
    lang = ud_map[args.evaldata_name]
    assert(lang in embs_map)
    print('POS tagging in {} ...'.format(lang))
    train_data, dev_data, test_data = CONLLU_loader.loadCONLLUFromDir(args.evaldata_path)
    #train_data, dev_data, test_data = train_data[:500], dev_data[:30], test_data[:30]
    pos_tagging_bilstm.evalEmbed(embs_map[lang], train_data, dev_data, test_data, lang, cuda) 

  if args.evaldata_name == 'semeval16t5':
    train_data, dev_data, test_data = semeval16t5_loader.loadData(args.evaldata_path)
    sentiment_analysis.evalEmbed(embs_map['en'], train_data['en']['subtask1']['restaurant'],
                                           dev_data['en']['subtask1']['restaurant'], 
                                           test_data['en']['subtask1']['restaurant'], 'en')


if __name__ == '__main__':
  args = config.parse_args()
  main(args)
