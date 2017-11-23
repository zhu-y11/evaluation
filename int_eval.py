# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Intrinsic Evaluation for word embeddings
@Author Yi Zhu
Upated 21/11/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import os

import torch

import config
import emb_loader
from word_similarity import simlex_calc
from word_similarity import wordsim_calc
from word_similarity import simverb_calc
from word_similarity import men_calc
from word_similarity import rareword_calc
from word_similarity import semeval17_t2


def main(args):
  print('loading word embeddings {} ...'.format(args.embedding_model[0]))
  embs_map = emb_loader.loadEmbeds(args.embedding_model,
                               args.embedding_dir_path,
                               args.embedding_file_name,
                               args.lower_case)
  assert(len(embs_map['en'][0]) == embs_map['en'][1].size()[0])

  loadDataAndEval(args, embs_map)


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
    # whether use transformed (crosslingual) embeddings
    #semeval17_t2.transform(embs_map)
    semeval17_t2.semEval17T2Calc(args.evaldata_path, embs_map, args.lower_case)


if __name__ == '__main__':
  args = config.parse_args()
  main(args)
