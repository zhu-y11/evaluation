# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Multilingual Sentiment Analysis 
@Author Yi Zhu
Upated 12/01/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import random
from random import shuffle
from tqdm import tqdm
import pdb
import string

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)
random.seed(1234)


class BiLSTM_SentimentAnalyzer(nn.Module):
  def __init__(self, word2idx, idx2word, lb2idx, idx2lb, 
               pre_embeds, 
               hidden_dim, num_layers = 2, bidirectional = True, mlpin_dim = 600, mlph1_dim = 800, mlph2_dim = 400):
    super(BiLSTM_SentimentAnalyzer, self).__init__()
    self.word2idx = word2idx
    self.idx2word = idx2word
    self.lb2idx = lb2idx
    self.idx2lb = idx2lb
    self.vocab_size = len(word2idx)
    self.target_size = len(lb2idx)
    
    # word embedding dim = pre-trained emb dim
    self.embed_dim = pre_embeds[1].size()[1]
    self.word_embeds = self.initEmbeds(pre_embeds)

    self.dropout = nn.Dropout(p = 0.4)

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional 
    self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, 
                        num_layers = self.num_layers, bidirectional = self.bidirectional)
    self.hidden = self.init_hidden()

    self.mlpintrans = nn.Linear(self.hidden_dim + 2 * self.embed_dim, mlpin_dim)
    self.mlp_i2h1 = nn.Linear(mlpin_dim, mlph1_dim)
    self.mlp_h12h2 = nn.Linear(mlph1_dim, mlph2_dim)
    self.mlp_h22o = nn.Linear(mlph2_dim, self.target_size)


  def initEmbeds(self, pre_embeds):
    print('initializing embedding layers ...')
    # initialize randomly
    embeds = nn.Embedding(self.vocab_size, self.embed_dim)
    for w, idx in tqdm(self.word2idx.items(), total = self.vocab_size):
      # if word not in pre-trained embeddings
      if w not in pre_embeds[0]:
        continue
      # word in the pretrained embeddings
      embeds.weight.data[idx] = pre_embeds[1][pre_embeds[0].index(w)]
    return embeds


  def init_hidden(self):
    return (autograd.Variable(torch.randn(self.num_layers * (2 if self.bidirectional else 1), 1, self.hidden_dim // 2)),
            autograd.Variable(torch.randn(self.num_layers * (2 if self.bidirectional else 1), 1, self.hidden_dim // 2)))


  def forward(self, sent, eas): 
    self.hidden = self.init_hidden()

    embeds = self.word_embeds(sent).view(len(sent), 1, -1)
    eas_embeds = self.word_embeds(eas).view(eas.size()[0], -1)
    if self.training:
      embeds = self.dropout(embeds)
      eas_embeds = self.dropout(eas_embeds)

    lstm_out, self.hidden = self.lstm(embeds)
    stack_lstm_out = lstm_out[-1].repeat(eas.size()[0], 1)

    h_eas_cat = torch.cat((stack_lstm_out, eas_embeds), 1)
    
    mlp_in = F.tanh(self.mlpintrans(h_eas_cat))
    h1 = F.relu(self.mlp_i2h1(mlp_in))
    h2 = F.relu(self.mlp_h12h2(h1))
    os = self.mlp_h22o(h2)
    prob_os = F.log_softmax(os)
    return prob_os


def evalEmbed(embs_tuple, train_data, dev_data, test_data, lang, bs = 50, epoch = 20, hidden_dim = 600, report_every = 5):
  word2idx, idx2word, lb2idx, idx2lb = getVocabAndLabel(train_data, dev_data, test_data)
  analyzer = BiLSTM_SentimentAnalyzer(word2idx, idx2word, lb2idx, idx2lb, embs_tuple, hidden_dim)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(analyzer.parameters(), lr = 0.001) 
  """ 
    [
      [
        Tensor([w1_idx, w2_idx, ... wn_idx]),
        Tensor(
          [e1_idx, a1_idx],
          [e2_idx, a2_idx],
          ...
        ),
        Tensor(
          [l1_idx, l2_idx, ...]
        )
      ],
      ...
    ]
  """
  train_seqs = prepSeqs(train_data, word2idx, lb2idx)
  dev_seqs = prepSeqs(dev_data, word2idx, lb2idx)
  test_seqs = prepSeqs(test_data, word2idx, lb2idx)
  best_acc_dev = .0
  corbest_acc_test = .0
  for i in range(epoch):
    # shuflle training data
    shuffle(train_seqs)
    n_batch = int(len(train_seqs) / bs) if len(train_seqs) % bs == 0 else int(len(train_seqs) / bs) + 1
    for j in range(n_batch):
      batch_train = train_seqs[j * bs: (j + 1) * bs]
      total_loss_train, acc_train = train(batch_train, analyzer, criterion, optimizer) 
      if j % report_every == 0:
        _, acc_dev = test(dev_seqs, analyzer, criterion)
        _, acc_test = test(test_seqs, analyzer, criterion)
        print('current dev_acc = {:.3f}%, current test_acc = {:.3f}%'.format(acc_dev, acc_test))
        if acc_dev > best_acc_dev:
          best_acc_dev = acc_dev
          corbest_acc_test = acc_test 
          torch.save(analyzer, 'best_analyzer_{}.model'.format(lang))
      print("epoch {}, batch {}/{}\nloss = {:.5f}  train_acc = {:.3f}%\nbest dev_acc = {:.3f}%  corresponding test_acc = {:.3f}%".format(
            i + 1, j + 1, n_batch, total_loss_train / len(batch_train), acc_train, best_acc_dev, corbest_acc_test))


def getVocabAndLabel(train_data, dev_data, test_data):
  vocab = []
  eas = []
  labels = []
  for sent, ops in train_data.items():
    # remove punctuations
    vocab.extend([word for word in sent.strip().split(' ') if word not in string.punctuation])
    for op in ops:
      vocab.extend([x.lower() for x in op['category'].split('#')])
      labels.append(op['polarity']) 
  for sent, ops in dev_data.items():
    for op in ops:
      labels.append(op['polarity']) 
  for sent, ops in test_data.items():
    for op in ops:
      labels.append(op['polarity']) 
  vocab.append('unk')
  vocab = set(vocab)
  labels = set(labels)
  tok2idx = dict(zip(vocab, range(len(vocab))))
  idx2tok = dict(zip(range(len(vocab)), vocab))
  label2idx = dict(zip(labels, range(len(labels))))
  idx2label = dict(zip(range(len(labels)), labels))
  return tok2idx, idx2tok, label2idx, idx2label


def prepSeq(seq, to_ix):
  idxs = [to_ix[w] if w in to_ix else to_ix['unk'] for w in seq]
  tensor = torch.LongTensor(idxs)
  return autograd.Variable(tensor)


def prepSeqs(data, word2idx, lb2idx):
  new_data = []
  for sent, ops in data.items():
    word_seq = prepSeq(sent.strip().split(' '), word2idx)
    eas = []
    lbs = []
    for op in ops:
      ea = [word2idx[x.lower()] for x in op['category'].split('#')]
      #ea_seq = prepSeq(ea, word2idx)
      label = lb2idx[op['polarity']]
      #lb_seq = prepSeq([label], lb2idx)
      eas.append(ea)
      lbs.append(label)
    eas = autograd.Variable(torch.LongTensor(eas))
    lbs = autograd.Variable(torch.LongTensor(lbs))
    new_data.append([word_seq, eas, lbs])
  return new_data


def train(train_seqs, analyzer, criterion, optimizer):
  optimizer.zero_grad()
  analyzer.train()
  total_loss = torch.Tensor([0])
  acc = 0
  n_inst = 0
  for train_seq, ea_seq, target_seq in train_seqs:
    probs = analyzer(train_seq, ea_seq)
    loss = criterion(probs, target_seq)
    acc += (probs.max(1)[1] == target_seq).data[0]
    total_loss += loss.data
    n_inst += len(target_seq) 
    loss.backward()
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  # update minibatch
  optimizer.step()
  return total_loss, acc


def test(seqs, analyzer, criterion):
  analyzer.eval()
  total_loss = torch.Tensor([0])
  acc = 0
  n_inst = 0
  for seq, ea_seq, target_seq in seqs:
    probs = analyzer(seq, ea_seq)
    loss = criterion(probs, target_seq)
    acc += (probs.max(1)[1] == target_seq).data[0]
    total_loss += loss.data
    n_inst += len(target_seq) 
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  return total_loss, acc
  
