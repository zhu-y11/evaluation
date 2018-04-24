# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Util scripts for word similarity task
@Author Yi Zhu
Upated 04/24/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.nn.functional as F

import metrics



def calcSpearmanr(emb_vocab, emb_vectors, word_pairs, sims):
  emb_sims = torch.Tensor(sims.size()).zero_()
  for i, (w1, w2) in enumerate(word_pairs):
    if w1 not in emb_vocab or w2 not in emb_vocab:
      emb_sims[i] = 0
      continue
    emb_sims[i] = F.cosine_similarity(emb_vectors[emb_vocab.index(w1)].view(1, -1), emb_vectors[emb_vocab.index(w2)].view(1, -1))[0] 
  return metrics.spearmanr(emb_sims, sims)


def calcBiSpearmanr(lang1_emb_vocab, lang1_emb_vectors,
                    lang2_emb_vocab, lang2_emb_vectors,
                    word_pairs, sims):
  """
   calculate bilingual crosslingual spearmanr
  """
  emb_sims = torch.Tensor(sims.size()).zero_()
  for i, (w1, w2) in enumerate(word_pairs):
    if w1 not in lang1_emb_vocab or w2 not in lang2_emb_vocab:
      emb_sims[i] = 0
      continue
    # TODO:
    # Need to add cross lingual embeddings, now just use 2 monolingual embeddings
    #
    emb_sims[i] = F.cosine_similarity(lang1_emb_vectors[lang1_emb_vocab.index(w1)].view(1, -1), 
                                      lang2_emb_vectors[lang2_emb_vocab.index(w2)].view(1, -1))[0] 
  return metrics.spearmanr(emb_sims, sims)


def findIntersec(vocab, vectors, word_pairs, sims):
  emb_idx = []
  data_idx = []
  for i, (w1, w2) in enumerate(word_pairs):
    if w1 in vocab and w2 in vocab:
      emb_idx.extend([vocab.index(w1), vocab.index(w2)])
      data_idx.append(i)
  emb_idx.sort()
  inter_vocab = [vocab[i] for i in emb_idx]
  inter_vectors = vectors.index_select(0, torch.LongTensor(emb_idx))
  inter_word_pairs = [word_pairs[i] for i in range(len(word_pairs)) if i in data_idx]
  inter_sims = sims.index_select(0, torch.LongTensor(data_idx))
  return inter_vocab, inter_vectors, inter_word_pairs, inter_sims


def findBiIntersec(lang1_vocab, lang1_vectors, 
                   lang2_vocab, lang2_vectors,
                   word_pairs, sims):
  lang1_emb_idx = []
  lang2_emb_idx = []
  data_idx = []
  for i, (w1, w2) in enumerate(word_pairs):
    if w1 in lang1_vocab and w2 in lang2_vocab:
      lang1_emb_idx.append(lang1_vocab.index(w1))
      lang2_emb_idx.append(lang2_vocab.index(w2))
      data_idx.append(i)
  lang1_emb_idx.sort()
  lang2_emb_idx.sort()
  lang1_inter_vocab = [lang1_vocab[i] for i in lang1_emb_idx]
  lang1_inter_vectors = lang1_vectors.index_select(0, torch.LongTensor(lang1_emb_idx))
  lang2_inter_vocab = [lang2_vocab[i] for i in lang2_emb_idx]
  lang2_inter_vectors = lang2_vectors.index_select(0, torch.LongTensor(lang2_emb_idx))
  inter_word_pairs = [word_pairs[i] for i in data_idx]
  inter_sims = sims.index_select(0, torch.LongTensor(data_idx))
  return lang1_inter_vocab, lang1_inter_vectors, lang2_inter_vocab, lang2_inter_vectors, inter_word_pairs, inter_sims

