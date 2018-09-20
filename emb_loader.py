# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Embedding Loader
@Author Yi Zhu
Upated 04/24/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import os
from tqdm import tqdm
import pickle
import subprocess
from gensim.models.keyedvectors import KeyedVectors
from word_similarity import semeval17_t2

import torch

import logging
logger = logging.getLogger(__name__)


def loadEmbed(emb_path, lower_case):
  emb_file_name = os.path.basename(emb_path)
  if '.txt' in emb_file_name:
    logger.debug('loading TEXT embeddings...')
    return loadTextEmd(emb_path, lower_case)


def loadTextEmd(emb_path, lower_case):
  base_emb_path = emb_path[:emb_path.rfind('.')]
  logger.debug('base embedding path: {}'.format(base_emb_path))
  # found the pytorch format
  if os.path.isfile(base_emb_path + '.pth') and os.path.isfile(base_emb_path + '.vocab'):
    return loadTorchEmd(base_emb_path, lower_case)

  logger.info('==> File not found, preparing, be patient')
  # pytorch format not found, read from txt file
  # and create tensors for word vectors
  with open(emb_path, 'r') as fin:
    f_line = fin.readline().rstrip().split()
  if len(f_line) == 2 and int(f_line[1]) > 0:
    headed = True
    # headed embedding
    """
      Embeddings with head format:

      vocab_size embedding_dim
      word1 d1 d2 ... dn
      word2 d1 d2 ... dn
      ...
      (space delimited)
    """
    count, emb_dim = list(map(int, f_line))
  else:
    emb_dim = len(f_line) - 1
    p = subprocess.Popen(['wc', '-l', embedding_file_path + '.txt'], 
                          stdout = subprocess.PIPE, 
                          stderr = subprocess.PIPE)
    count = int(p.communicate()[0].decode('utf-8').strip().split()[0])
  vocab = [None] * (count)
  vectors = torch.zeros(count, emb_dim)

  with open(emb_path, 'r') as fin:
    if headed:
      fin.readline()
    idx = 0
    for line in tqdm(fin, total = count):
      line = line.rstrip().split(' ')
      vocab[idx] = line[0]
      vectors[idx] = torch.Tensor(list(map(float, line[1:])))
      vectors[idx] = vectors[idx] / vectors[idx].norm()
      idx += 1
  with open(base_emb_path + '.vocab','w') as f:
    for word in vocab:
      f.write(word + '\n')
  torch.save(vectors, base_emb_path + '.pth')
  vocab = [v.lower() if lower_case else v for v in vocab]
  assert(len(vocab) == vectors.size()[0])
  return (vocab, vectors)


def loadTorchEmd(base_emb_path, lower_case):
    logger.info('==> Vocab and pth file found, loading to memory...')
    with open(base_emb_path + '.vocab') as f:
      vocab = f.read().rstrip().split('\n')
    vocab = [v.lower() if lower_case else v for v in vocab]
    vectors = torch.load(base_emb_path + '.pth')
    vectors = torch.div(vectors, vectors.norm(p = 2, dim = 1).view(-1, 1))
    return (vocab, vectors)


def loadPklEmbed(emb_dir_path, emb_file_name, lower_case):
  """
  After unpickle, format:

  [
    [word1, word2, ..., wordn],
    [
      [d1, d2, ..., dm],
      [d1, d2, ..., dm],
      ...
    ]
  ]
  """
  emb_file_path = os.path.join(emb_dir_path, emb_file_name)
  emb_files = pickle.load(open(emb_file_path + '.pkl', 'rb'), encoding = 'latin1') 
  vocab = [v.lower() if lower_case else v for v in emb_files[0]]
  count = len(vocab)
  emb_dim = len(emb_files[1][0])
  vectors = torch.zeros(count, emb_dim)  
  for i, vec in tqdm(enumerate(emb_files[1]), total = count):
    vectors[i] = torch.Tensor(list(map(float, vec)))
    vectors[i] = vectors[i] / vectors[i].norm()
  return [vocab, vectors]
   
 
def loadBinEmbed(emb_dir_path, emb_file_name, lower_case):
  emb_file_path = os.path.join(emb_dir_path, emb_file_name)
  try:
    return loadHeadEmbed(emb_dir_path, emb_file_name, lower_case)
  except:
    # no pre-saved model, no .txt model, must have .bin model
    model = KeyedVectors.load_word2vec_format(emb_file_path + '.bin', binary = True, encoding='utf-8', unicode_errors='ignore')
    model.save_word2vec_format(emb_file_path + '.txt', binary = False)
  return loadHeadEmbed(emb_dir_path, emb_file_name, lower_case)


def loadNumberbatchEmbed(emb_dir_path, emb_file_name, langs, lower_case):
  """
    Numberbatch format
    vocab_size embedding_dim
    word1 d1 d2 ... dn
    word2 d1 d2 ... dn
    ...
    (space delimited)
  """
  emb_file_path = os.path.join(emb_dir_path, emb_file_name)
  embs_map = {}
  for lang in langs[:]:
    if os.path.isfile(emb_file_path + '.{}.pth'.format(lang)) and\
       os.path.isfile(emb_file_path + '.{}.vocab'.format(lang)):
      print('==> {} found, loading to memory'.format(lang))
      with open(emb_file_path + '.{}.vocab'.format(lang), 'r') as f:
        vocab = f.read().rstrip().split('\n')
      vocab = [v.lower() if lower_case else v for v in vocab]
      vectors = torch.load(emb_file_path + '.{}.pth'.format(lang))
      embs_map[lang] = [vocab, vectors]
      langs.remove(lang)
  if not langs:
    return embs_map

  print('==> File not found, preparing, be patient')
  count, emb_dim = list(map(int, open(emb_file_path + '.txt').readline().strip().split(' '))) 
  with open(emb_file_path + '.txt', 'r') as f:
    f.readline()
    for line in tqdm(f, total = count):
      line = line.rstrip().split(' ')
      _, _, lang, word = line[0].split('/')
      if lang not in langs:
        continue
      word = word.lower() if lower_case else word
      vec = list(map(float, line[1:]))
      if lang not in embs_map:
        embs_map[lang] = [[word], [vec]]
      else:
        embs_map[lang][0].append(word)
        embs_map[lang][1].append(vec)
  for lang in embs_map:
    with open(emb_file_path + '.{}.vocab'.format(lang), 'w') as f:
      for word in embs_map[lang][0]:
        f.write(word + '\n')
    embs_map[lang][1] = torch.Tensor(embs_map[lang][1])
    torch.save(embs_map[lang][1], emb_file_path + '.{}.pth'.format(lang))

  return embs_map



if __name__ == '__main__':
  emb_dir_path = sys.argv[1]
  emb_file_name = sys.argv[2]
  loadBinEmbed(emb_dir_path, emb_file_name, True)
