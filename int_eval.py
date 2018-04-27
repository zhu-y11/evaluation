# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Intrinsic Evaluation for word embeddings
@Author Yi Zhu
Upated 04/24/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

import config
import emb_loader
from word_similarity import calc_wordsim

import logging
logger = logging.getLogger(__name__)


def word_similarity(args, test_data_dir):
  logger.info('Calculating word similarity...')
  for i, emb_path in enumerate(args.emb_path):
    lang = args.lang[i]
    logger.info('Loading word embeddings: {}'.format(os.path.basename(emb_path)))
    vocab, emb = emb_loader.loadEmbed(emb_path, args.lower_case)
    calc_wordsim.eval_word_similarity(lang, test_data_dir, vocab, emb, args.lower_case) 


def loadDataAndEval(args, embs_map):
  print('loading {} dataset from\n{} ...'.format(args.evaldata_name, args.evaldata_path))
  if args.evaldata_name == 'simlex999':
    r, inter_r = simlex_calc.simLexCalc(args.evaldata_path, embs_map['en'][0], embs_map['en'][1], args.lower_case)
    print('{} {:.5f}'.format(r[0], r[1]))
    print('{} {:.5f}'.format(inter_r[0], inter_r[1]))
  if args.evaldata_name == 'multi_simlex999':
    simlex_calc.multSimLexCalc(args.evaldata_path, embs_map, args.lower_case)
  if args.evaldata_name.startswith('wordsim353'):
    if args.evaldata_name == 'wordsim353':
      r, inter_r = wordsim_calc.wordSimCalc(args.evaldata_path, embs_map['en'][0], embs_map['en'][1], args.lower_case)
      print('{} {:.5f}'.format(r[0], r[1]))
      print('{} {:.5f}'.format(inter_r[0], inter_r[1]))
    elif args.evaldata_name.endswith('rel') or args.evaldata_name.endswith('sim'):
      r, inter_r = wordsim_calc.wordSimCalcRelSim(args.evaldata_path, embs_map['en'][0], embs_map['en'][1], args.lower_case)
      print('{} {:.5f}'.format(r[0], r[1]))
      print('{} {:.5f}'.format(inter_r[0], inter_r[1]))
  if args.evaldata_name.startswith('multi_wordsim353'):
    if args.evaldata_name.endswith('rel') or args.evaldata_name.endswith('sim'):
      wordsim_calc.multWordSimCalcRelSim(args.evaldata_path, embs_map, args.lower_case)
    else:
      wordsim_calc.multWordSimCalc(args.evaldata_path, embs_map, args.lower_case)
  if args.evaldata_name == 'simverb3500':
    r, inter_r = simverb_calc.simVerbCalc(args.evaldata_path, embs_map['en'][0], embs_map['en'][1], args.lower_case)
    print('{} {:.5f}'.format(r[0], r[1]))
    print('{} {:.5f}'.format(inter_r[0], inter_r[1]))
  if args.evaldata_name == 'men3000':
    r, inter_r = men_calc.menCalc(args.evaldata_path, embs_map['en'][0], embs_map['en'][1], args.lower_case)
    print('{} {:.5f}'.format(r[0], r[1]))
    print('{} {:.5f}'.format(inter_r[0], inter_r[1]))
  if args.evaldata_name == 'rareword':
    r, inter_r = rareword_calc.rarewordCalc(args.evaldata_path, embs_map['en'][0], embs_map['en'][1], args.lower_case)
    print('{} {:.5f}'.format(r[0], r[1]))
    print('{} {:.5f}'.format(inter_r[0], inter_r[1]))
  if args.evaldata_name == 'semeval17t2':
    semeval17_t2.semEval17T2Calc(args.evaldata_path, embs_map, args.lower_case)
