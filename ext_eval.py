# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Extrinsic Evaluation for word embeddings
@Author Yi Zhu
Upated 06/07/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

import config, emb_loader
from absa import semeval16t5_loader, sentiment_analysis

import logging
logger = logging.getLogger(__name__)

import pdb

def absa(args, data_path, sbt, slt, domain):
  logger.info('Loading ABSA Data...')
  train_data, dev_data, test_data = semeval16t5_loader.loadData(data_path)
  for i, emb_path in enumerate(args.emb_path):
    lang = args.lang[i]
    if lang not in train_data or lang not in dev_data or lang not in test_data:
      logger.debug('ABSA {} data does not exist, proceed to next embeddings'.format(lang))
      continue
    logger.info('Loading word embeddings: {}'.format(os.path.basename(emb_path)))
    vocab, emb = emb_loader.loadEmbed(emb_path, args.lower_case)
    logger.info('Lang {}, absa {} {} {}...'.format(lang, sbt, slt, domain))
    sentiment_analysis.evalEmbed(vocab, emb, 
                                train_data[lang][sbt][domain],
                                dev_data[lang][sbt][domain], 
                                test_data[lang][sbt][domain], 
                                lang, slt, args.lower_case, args.cuda)


def main(args):
  args.cuda = args.cuda if torch.cuda.is_available() else False
  print('loading word embeddings {} ...'.format(args.embedding_model[0]))
  embs_map = {}
  embs_map = emb_loader.loadEmbeds(args.embedding_model,
                               args.embedding_dir_path,
                               args.embedding_file_name,
                               args.lower_case)
  if 'vi' in embs_map:
    for i in range(len(embs_map['vi'][0])):
      print(embs_map['vi'][0][i])
      embs_map['vi'][0][i] = ' '.join(embs_map['vi'][0][i].split('_'))
      print(embs_map['vi'][0][i])  
  loadDataAndEval(args, embs_map)


def loadDataAndEval(args, embs_map):
  #print('loading {} dataset from\n{} ...'.format(args.evaldata_name, args.evaldata_path))
  if args.task == 'pos_tagging':
    lang = ud_map[args.evaldata_name]
    assert(lang in embs_map)
    print('POS tagging in {} ...'.format(lang))
    train_data, dev_data, test_data = CONLLU_loader.loadCONLLUFromDir(args.evaldata_path)
    #train_data, dev_data, test_data = train_data[:500], dev_data[:30], test_data[:30]
    pos_tagging_bilstm.evalEmbed(embs_map[lang], train_data, dev_data, test_data, lang, args.cuda) 

  if args.evaldata_name == 'semeval16t5':
    train_data, dev_data, test_data = semeval16t5_loader.loadData(args.evaldata_path)
    sentiment_analysis.evalEmbed(embs_map['en'], train_data['en']['subtask1']['restaurant'],
                                                 dev_data['en']['subtask1']['restaurant'], 
                                                 test_data['en']['subtask1']['restaurant'], 'en', args.cuda)
