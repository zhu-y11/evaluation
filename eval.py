#-*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  06/07/2018 
Main entry of evaluation scripts
"""

#************************************************************
# Imported Libraries
#************************************************************
from config import parse_args
from int_eval import word_similarity
from ext_eval import absa

import logging
logger = logging.getLogger()
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


def main(args):
  for i, t in enumerate(args.task):
    test_data_path = args.evaldata_path[i]
    # word similarity
    if t == 'word_similarity':
      word_similarity(args, test_data_path)
    # aspect-based sentiment analysis
    if t.startswith('absa'):
      t_vec = t.split('_')
      # subtask
      sbt = t_vec[1]
      # slot
      slt = t_vec[2]
      # data domain
      domain = t_vec[3]
      absa(args, test_data_path, sbt, slt, domain)


if __name__ == '__main__':
  args = parse_args() 
  main(args)
