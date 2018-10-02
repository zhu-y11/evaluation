# -*- coding: latin-1 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  09/22/2018
Generate word list
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import os
import codecs

import pdb

data_dir = sys.argv[1]
word_set = []
# for turkish 
#with codecs.open(in_file, 'r', encoding = 'latin-1') as fin:
for root, dirs, files in os.walk(data_dir):
  for f in files:
    if not f.endswith('.txt'):
      continue
    with open(os.path.join(root, f), 'r') as fin:
      for line in fin:
        linevec = line.strip().lower().split()[:2]
        if os.path.basename(f).startswith('7.card660.en'):
          linevec = [w.strip().split('_') for w in linevec]
          linevec = [w for w_seq in linevec for w in w_seq]
        word_set.extend(linevec)
word_set = list(set(word_set))
print('\n'.join(word_set))
