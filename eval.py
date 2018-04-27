#-*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  02/24/2018 
Main entry of evaluation scripts
"""

#************************************************************
# Imported Libraries
#************************************************************
from config import parse_args
from int_eval import word_similarity

import logging
logger = logging.getLogger()
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)


def main(args):
  for i, t in enumerate(args.task):
    if t == 'word_similarity':
      test_data_path = args.evaldata_path[i]
      word_similarity(args, test_data_path)


if __name__ == '__main__':
  args = parse_args() 
  main(args)
