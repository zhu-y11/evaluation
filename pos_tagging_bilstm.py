# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
POS Tagging with Bi-LSTM
@Author Yi Zhu
Upated 08/12/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import random
from random import shuffle
from tqdm import tqdm
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)
random.seed(1234)


# Create model
class BiLSTM_POSTagger(nn.Module):
  def __init__(self, word2idx, idx2word, tag2idx, idx2tag, 
               pre_embeds, hidden_dim, cuda):
    super(BiLSTM_POSTagger, self).__init__()
    self.word2idx = word2idx
    self.idx2word = idx2word
    self.tag2idx = tag2idx
    self.idx2tag = idx2tag
    self.vocab_size = len(word2idx)
    self.target_size = len(tag2idx)
    self.use_cuda = cuda

    self.embed_dim = pre_embeds[1].size()[1]
    self.hidden_dim = hidden_dim

    self.word_embeds = self.initEmbeds(pre_embeds)

    self.lstm = nn.LSTM(self.embed_dim, hidden_dim // 2, 
                        num_layers = 2, bidirectional = True)
    
    self.dropout = nn.Dropout(p = 0.4)
    # Maps the output of the LSTM into tag space.
    self.hidden2tag = nn.Linear(hidden_dim, self.target_size)

    self.hidden = self.init_hidden()


  def initEmbeds(self, pre_embeds):
    print('initializing embedding layers ...')
    embeds = nn.Embedding(self.vocab_size, self.embed_dim)
    for w, idx in tqdm(self.word2idx.items(), total = self.vocab_size):
      if w not in pre_embeds[0]:
        continue
      embeds.weight.data[idx] = pre_embeds[1][pre_embeds[0].index(w)]
    return embeds


  def init_hidden(self):
      hidden = autograd.Variable(torch.randn(4, 1, self.hidden_dim // 2)),
               autograd.Variable(torch.randn(4, 1, self.hidden_dim // 2))
    if self.use_cuda:
      return (hidden[0].cuda(), hidden[1].cuda())
    else:
      return hidden


  def forward(self, sentence):     
    self.hidden = self.init_hidden()
    
    embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
    if self.training:
      embeds = self.dropout(embeds)

    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
    if self.training:
      lstm_out = self.dropout(lstm_out)

    tag_space = self.hidden2tag(lstm_out)
    tag_scores = F.log_softmax(tag_space).view(len(sentence), -1)
    return tag_scores    



def evalEmbed(embs_tuple, train_data, dev_data, test_data, lang, cuda, bs = 100, epoch = 10, hidden_dim = 600, report_every = 5):
  word2idx, idx2word, tag2idx, idx2tag = getVocabAndTag(train_data, dev_data, test_data)
  tagger = BiLSTM_POSTagger(word2idx, idx2word, tag2idx, idx2tag, embs_tuple, hidden_dim, cuda)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(tagger.parameters(), lr = 0.01) 
  
  """
    [
      [
        Tensor([w1_idx, w2_idx, ... wn_idx]),
        Tensor([t1_idx, t2_idx, ... tn_idx])
      ],
      ...
    ]
  """
  train_seqs = prepSeqs(train_data, word2idx, tag2idx)
  dev_seqs = prepSeqs(dev_data, word2idx, tag2idx)
  test_seqs = prepSeqs(test_data, word2idx, tag2idx)
  best_acc_dev = .0
  corbest_acc_test = .0

  if cuda:
    tagger.cuda()
    criterion = criterion.cuda()
    
  for i in range(epoch):
    # shuflle training data
    shuffle(train_seqs)
    n_batch = int(len(train_seqs) / bs) if len(train_seqs) % bs == 0 else int(len(train_seqs) / bs) + 1
    for j in range(n_batch):
      batch_train = train_seqs[j * bs: (j + 1) * bs]
      total_loss_train, acc_train = train(batch_train, tagger, criterion, optimizer, cuda) 
      if j % report_every == 0:
        _, acc_dev = test(dev_seqs, tagger, criterion, cuda)
        _, acc_test = test(test_seqs, tagger, criterion, cuda)
        print('current dev_acc = {:.3f}%, current test_acc = {:.3f}%'.format(acc_dev, acc_test))
        if acc_dev > best_acc_dev:
          best_acc_dev = acc_dev
          corbest_acc_test = acc_test 
          torch.save(tagger, 'best_tagger_{}.model'.format(lang))
      print("epoch {}, batch {}/{}\nloss = {:.5f}  train_acc = {:.3f}%\nbest dev_acc = {:.3f}%  corresponding test_acc = {:.3f}%".format(
            i + 1, j + 1, n_batch, total_loss_train / len(batch_train), acc_train, best_acc_dev, corbest_acc_test))


def getVocabAndTag(train_data, dev_data, test_data):
  vocab = []
  tags = []
  for sent in train_data:
    for line in sent:
      vocab.append(line[1].lower())
      tags.append(line[3])
  for sent in dev_data:
    for line in sent:
      tags.append(line[3])
  for sent in test_data:
    for line in sent:
      tags.append(line[3])
  vocab.append('unk')
  vocab = set(vocab)
  tags = set(tags)
  tok2idx = dict(zip(vocab, range(len(vocab))))
  idx2tok = dict(zip(range(len(vocab)), vocab))
  tag2idx = dict(zip(tags, range(len(tags))))
  idx2tag = dict(zip(range(len(tags)), tags))
  return tok2idx, idx2tok, tag2idx, idx2tag


def prepSeq(seq, to_ix):
  idxs = [to_ix[w] if w in to_ix else to_ix['unk'] for w in seq]
  tensor = torch.LongTensor(idxs)
  return autograd.Variable(tensor)


def prepSeqs(data, word2idx, tag2idx):
  new_data = []
  for sent in data:
    words, tags = zip(*[[line[1].lower(), line[3]] for line in sent])
    word_seq = prepSeq(words, word2idx)
    tag_seq = prepSeq(tags, tag2idx)
    new_data.append([word_seq, tag_seq])
  return new_data


def train(train_seqs, tagger, criterion, optimizer, cuda):
  optimizer.zero_grad()
  tagger.train()
  total_loss = torch.cuda.FloatTensor([0]) if cuda else torch.FloatTensor([0])
  acc = 0
  n_inst = 0
  for train_seq, targets in train_seqs:
    if cuda:
      train_seq = train_seq.cuda()
      targets = targets.cuda()
    tag_scores = tagger(train_seq)
    acc += (tag_scores.max(1)[1] == targets).sum().data[0]
    n_inst += tag_scores.size()[0]
    loss = criterion(tag_scores, targets)
    loss.backward()
    total_loss += loss.data
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  # update minibatch
  optimizer.step()
  return total_loss, acc


def test(seqs, tagger, criterion, cuda):
  tagger.eval()
  total_loss = torch.cuda.FloatTensor([0]) if cuda else torch.FloatTensor([0])
  acc = 0
  n_inst = 0
  for seq, targets in seqs:
    if cuda:
      seq = seq.cuda()
      targets = targets.cuda()
    tag_scores = tagger(seq)
    acc += (tag_scores.max(1)[1] == targets).sum().data[0]
    n_inst += tag_scores.size()[0]
    loss = criterion(tag_scores, targets)
    total_loss += loss.data    
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  return total_loss, acc
