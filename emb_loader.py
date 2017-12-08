# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Embedding Loader
@Author Yi Zhu
Upated 22/11/2017 
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


def loadEmbeds(emb_model, emb_dir_path, emb_file_name, lower_case):
  embs_map = {}
  if emb_model[0] == 'glove':
    for i, lang in enumerate(emb_model[1:]):
      print('loading {} {} ...'.format(emb_model[0], emb_model[i + 1]))
      embs_map[lang] = loadStdEmbed(emb_dir_path, emb_file_name[i], lower_case)
  if emb_model[0] == 'fasttext':
    for i, lang in enumerate(emb_model[1:]):
      print('loading {} {} ...'.format(emb_model[0], emb_model[i + 1]))
      embs_map[lang] = loadHeadEmbed(emb_dir_path, emb_file_name[i], lower_case)
    # whether use transformed (crosslingual) embeddings
    #semeval17_t2.transform(embs_map)    
  if emb_model[0] == 'polyglot':
    for i, lang in enumerate(emb_model[1:]):
      print('loading {} {} ...'.format(emb_model[0], emb_model[i + 1]))
      embs_map[lang] = loadPklEmbed(emb_dir_path, emb_file_name[i], lower_case)
  if emb_model[0] == 'word2vec':
    for i, lang in enumerate(emb_model[1:]):
      print('loading {} {} ...'.format(emb_model[0], emb_model[i + 1]))
      embs_map[lang] = loadBinEmbed(emb_dir_path, emb_file_name[i], lower_case)
  if emb_model[0] == 'numberbatch':
    print('loading {} ...'.format(emb_model[0]))
    embs_map = loadNumberbatchEmbed(emb_dir_path, emb_file_name[0], emb_model[1:], lower_case)
  return embs_map


def loadStdEmbed(embedding_dir_path, embedding_file_name, lower_case):
  """
    Standard Embeddings format:

    word1 d1 d2 ... dn
    word2 d1 d2 ... dn
    (space delimited)
    ...
  """
  embedding_file_path = os.path.join(embedding_dir_path, embedding_file_name)
  if os.path.isfile(embedding_file_path + '.pth') and os.path.isfile(embedding_file_path + '.vocab'):
    print('==> File found, loading to memory')
    with open(embedding_file_path + '.vocab') as f:
      vocab = f.read().rstrip().split('\n')
    vocab = [v.lower() if lower_case else v for v in vocab]
    vectors = torch.load(embedding_file_path + '.pth')
    vectors = torch.div(vectors, vectors.norm(p = 2, dim = 1).view(-1, 1))
    return [vocab, vectors]
  # saved file not found, read from txt file
  # and create tensors for word vectors
  print('==> File not found, preparing, be patient')
  embedding_dim = len(open(embedding_file_path + '.txt', 'r').readline().rstrip().split(' ')[1:])
  p = subprocess.Popen(['wc', '-l', embedding_file_path + '.txt'], 
                        stdout = subprocess.PIPE, 
                        stderr=subprocess.PIPE)
  count = int(p.communicate()[0].decode('utf-8').strip().split()[0])
  vocab = [None] * (count)
  vectors = torch.zeros(count, embedding_dim)
  with open(embedding_file_path + '.txt', 'r') as f:
    idx = 0
    for line in tqdm(f, total = count):
      line = line.rstrip().split(' ')
      vocab[idx] = line[0]
      vectors[idx] = torch.Tensor(list(map(float, line[1:])))
      vectors[idx] = vectors[idx] / vectors[idx].norm()
      idx += 1
  with open(embedding_file_path + '.vocab','w') as f:
    for word in vocab:
      f.write(word + '\n')
  torch.save(vectors, embedding_file_path + '.pth')
  vocab = [v.lower() if lower_case else v for v in vocab]
  return [vocab, vectors]


def loadHeadEmbed(emb_dir_path, emb_file_name, lower_case):
  """
    Embeddings with head format:

    vocab_size embedding_dim
    word1 d1 d2 ... dn
    word2 d1 d2 ... dn
    ...
    (space delimited)
  """
  emb_file_path = os.path.join(emb_dir_path, emb_file_name)
  if os.path.isfile(emb_file_path + '.pth') and os.path.isfile(emb_file_path + '.vocab'):
    print('==> File found, loading to memory')
    with open(emb_file_path + '.vocab') as f:
      vocab = f.read().rstrip().split('\n')
    vocab = [v.lower() if lower_case else v for v in vocab]
    vectors = torch.load(emb_file_path + '.pth')
    vectors = torch.div(vectors, vectors.norm(p = 2, dim = 1).view(-1, 1))
    return [vocab, vectors]
  # saved file not found, read from txt file
  # and create tensors for word vectors
  print('==> File not found, preparing, be patient')
  count, emb_dim = list(map(int, open(emb_file_path + '.txt').readline().strip().split(' ')))
  vocab = [None] * (count)
  vectors = torch.zeros(count, emb_dim)
  with open(emb_file_path + '.txt', 'r') as f:
    idx = 0
    f.readline()
    for line in tqdm(f, total = count):
      line = line.rstrip().split(' ')
      vocab[idx] = line[0]
      vectors[idx] = torch.Tensor(list(map(float, line[1:])))
      vectors[idx] = vectors[idx] / vectors[idx].norm()
      idx += 1
  with open(emb_file_path + '.vocab','w') as f:
    for word in vocab:
      f.write(word + '\n')
  torch.save(vectors, emb_file_path + '.pth')
  vocab = [v.lower() if lower_case else v for v in vocab]
  return [vocab, vectors]


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
    model = KeyedVectors.load_word2vec_format(emb_file_path + '.bin', binary = True)
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
