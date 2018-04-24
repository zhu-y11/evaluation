# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  04/24/2018
Normalize word similarity data
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys

in_file = sys.argv[1]
with open(in_file, 'r') as fin:
  for line in fin:
    #print(line.strip())
    # try to first split by ,
    linevec = line.strip().split(',')
    if len(linevec) < 3:
      # can not be split by ,
      # we split by space/tab
      linevec = line.strip().split()
      assert(len(linevec) >= 3)
    word1 = linevec[0].strip()
    word2 = linevec[1].strip()
    try:
      score = float(linevec[-1])
      print(word1, word2, score)
    except:
      continue
