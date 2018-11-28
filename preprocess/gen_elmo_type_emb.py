# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated 10/28/2018
Generate type level elmo embeddings
"""

#************************************************************
# Imported Libraries
#************************************************************
import h5py
import sys
import ast
import numpy as np

import pdb

# input hdf5 file
in_file = sys.argv[1]
base_in_file = in_file[:in_file.find('.txt')]

layers = ['0', '1', '2', '12', '012']
directions = ['bidi', 'forward', 'backward']

h5py_file = h5py.File(in_file, 'r')
key_list = list(h5py_file.keys())
sent2idx = ast.literal_eval(h5py_file.get(key_list[-1])[0])
idx2sent = {v: k for k, v in sent2idx.items()}
sent_n = len(key_list) - 1

for l in layers:
  l_num = int(l)
  for direction in directions:
    out_file = base_in_file + '.elmo.l{}.{}.txt'.format(l, direction)
    with open(out_file, 'w') as fout:
      dim = 512 if direction == 'forward' or direction == 'backward' else 1024
      fout.write('{} {}\n'.format(sent_n, dim))
      for i in range(sent_n):
        emb_i = h5py_file.get('{}'.format(i))
        # one layer for each vector
        assert(len(emb_i) == 3)         
        # only get emb for the last word
        if l == '12':
          # average l1 and l2
          emb_i = np.mean(emb_i[1:, -1], axis = 0)
        elif l == '012':
          # average l1, l2 and l3
          emb_i = np.mean(emb_i[:, -1], axis = 0)
        else:
          # a single layer
          emb_i = emb_i[l_num, -1]

        assert(emb_i.shape == (1024, ))
        if direction == 'forward':
          emb_i = emb_i[:512]
        elif direction == 'backward':
          emb_i = emb_i[512:]

        word = idx2sent['{}'.format(i)].strip().split()[-1]
        emb_list = [word] + [str(e) for e in emb_i]
        emb_str = ' '.join(emb_list)
        fout.write(emb_str + '\n')

