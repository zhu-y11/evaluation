# -*- coding: latin-1 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  04/24/2018
Normalize word similarity data set
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import codecs

import pdb

in_file = sys.argv[1]
score_pos = int(sys.argv[2])

header = True

#with open(in_file, 'r') as fin
# for turkish
with codecs.open(in_file, 'r', encoding = 'latin-1') as fin:
  if header:
    fin.readline()
  for line in fin:
    #print(line.strip())
    # try to first split by ,
    linevec = line.strip().split(';')
    if len(linevec) < 3:
      # can not be split by ,
      # we split by space/tab
      linevec = line.strip().split()
      assert(len(linevec) >= 3)
    word1 = linevec[1].strip()
    word2 = linevec[2].strip()
    try:
      score = float(linevec[score_pos - 1].replace(',','.'))
      print(word1, word2, score)
    except:
      continue
