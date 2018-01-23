# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Multilingual Sentiment Analysis 
@Author Yi Zhu
Upated 23/01/2018
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
  def __init__(self, word2idx, idx2word, ent2idx, idx2ent, asp2idx, idx2asp, lb2idx, idx2lb, pre_embeds, 
               hidden_dim = 200, ea_dim = 50, num_layers = 2, bidirectional = True, mlpin_dim = 100, mlph_dim = 60):
    super(BiLSTM_SentimentAnalyzer, self).__init__()
    self.word2idx = word2idx
    self.idx2word = idx2word
    self.ent2idx = ent2idx
    self.idx2ent = idx2ent
    self.asp2idx = asp2idx
    self.idx2asp = idx2asp
    self.lb2idx = lb2idx
    self.idx2lb = idx2lb
    self.vocab_size = len(word2idx)
    self.ent_size = len(ent2idx)
    self.asp_size = len(asp2idx)
    self.target_size = len(lb2idx)
    
    # word embedding dim = pre-trained emb dim
    self.embed_dim = pre_embeds[1].size()[1]
    self.word_embeds = self.initEmbeds(pre_embeds)

    # entity dim and aspect dim are the same
    self.ea_dim = ea_dim
    # initialize both randomly
    self.ent_embeds = nn.Embedding(self.ent_size, self.ea_dim)
    self.asp_embeds = nn.Embedding(self.asp_size, self.ea_dim)

    self.dropout = nn.Dropout(p = 0.4)

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional 
    self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, 
                        num_layers = self.num_layers, dropout = 0.4, bidirectional = self.bidirectional)
    self.hidden = self.init_hidden()

    self.mlpintrans = nn.Linear(self.hidden_dim + 2 * self.ea_dim, mlpin_dim)
    self.mlp_itoh = nn.Linear(mlpin_dim, mlph_dim)
    self.mlp_htoo = nn.Linear(mlph_dim, self.target_size)


  def initEmbeds(self, pre_embeds):
    print('initializing embedding layers ...')
    # initialize randomly
    embeds = nn.Embedding(self.vocab_size, self.embed_dim)
    for w, idx in tqdm(self.word2idx.items(), total = self.vocab_size):
      # skip words that are not in pre-trained embeddings
      if w not in pre_embeds[0]:
        continue
      # initialize words that are in the pretrained embeddings
      embeds.weight.data[idx] = pre_embeds[1][pre_embeds[0].index(w)]
    return embeds


  def init_hidden(self):
    return (autograd.Variable(torch.randn(self.num_layers * (2 if self.bidirectional else 1), 1, self.hidden_dim // 2)),
            autograd.Variable(torch.randn(self.num_layers * (2 if self.bidirectional else 1), 1, self.hidden_dim // 2)))


  def forward(self, sent, ents, asps): 
    self.hidden = self.init_hidden()

    embeds = self.word_embeds(sent).view(len(sent), 1, -1)
    ent_embeds = self.ent_embeds(ents).view(len(ents), -1)
    asp_embeds = self.ent_embeds(asps).view(len(asps), -1)
    if self.training:
      embeds = self.dropout(embeds)
      ent_embeds = self.dropout(ent_embeds)
      asp_embeds = self.dropout(asp_embeds)

    lstm_out, self.hidden = self.lstm(embeds)
    stack_lstm_out = lstm_out[-1].repeat(ents.size()[0], 1)

    h_eas_cat = torch.cat((stack_lstm_out, ent_embeds, asp_embeds), 1)
    
    mlp_in = F.tanh(self.mlpintrans(h_eas_cat))
    h = F.relu(self.mlp_itoh(mlp_in))
    os = self.mlp_htoo(h)
    prob_os = F.log_softmax(os, 1)
    return prob_os


def evalEmbed(embs_tuple, train_data, dev_data, test_data, lang, cuda, bs = 16, epoch = 20, report_every = 2):
  if cuda:
    torch.cuda.manual_seed(1234)
  word2idx, idx2word, \
  ent2idx, idx2ent,\
  asp2idx, idx2asp,\
  lb2idx, idx2lb = getVocabAndLabel(train_data, dev_data, test_data)
  analyzer = BiLSTM_SentimentAnalyzer(word2idx, idx2word, 
                                      ent2idx, idx2ent, 
                                      asp2idx, idx2asp,
                                      lb2idx, idx2lb, 
                                      embs_tuple)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(analyzer.parameters(), lr = 0.001) 

  if cuda:
    analyzer.cuda()

  ''' 
    [
      [
        Tensor([w1_idx, w2_idx, ... wn_idx]),
        Tensor([e1_idx, e2_idx, ..., em_idx]),
        Tensor([a1_idx, a2_idx, ..., am_idx]),
        Tensor([l1_idx, l2_idx, ..., lm_idx])
      ],
      ...
    ]
  '''
  train_seqs = prepSeqs(train_data, word2idx, ent2idx, asp2idx, lb2idx, cuda)
  dev_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, lb2idx, cuda)
  test_seqs = prepSeqs(test_data, word2idx, ent2idx, asp2idx, lb2idx, cuda)
  best_acc_dev = .0
  corbest_acc_test = .0
  # total batch number
  n_batch = int(len(train_seqs) / bs) if len(train_seqs) % bs == 0 else int(len(train_seqs) / bs) + 1
  for i in range(epoch):
    # shuflle training data
    shuffle(train_seqs)
    for j in range(n_batch):
      batch_train = train_seqs[j * bs: (j + 1) * bs]
      total_loss_train, acc_train = train(batch_train, analyzer, criterion, optimizer, cuda) 
      if j % report_every == 0:
        _, acc_dev = test(dev_seqs, analyzer, criterion, cuda)
        _, acc_test = test(test_seqs, analyzer, criterion, cuda)
        print('current dev_acc = {:.3f}%, current test_acc = {:.3f}%'.format(acc_dev, acc_test))
        if acc_dev > best_acc_dev:
          best_acc_dev = acc_dev
          corbest_acc_test = acc_test 
          #torch.save(analyzer, 'best_analyzer_{}.model'.format(lang))
      print("epoch {}, batch {}/{}\nloss = {:.5f}  train_acc = {:.3f}%\nbest dev_acc = {:.3f}%  corresponding test_acc = {:.3f}%".format(
            i + 1, j + 1, n_batch, total_loss_train / len(batch_train), acc_train, best_acc_dev, corbest_acc_test))


def getVocabAndLabel(train_data, dev_data, test_data):
  vocab = [] # vocabulary
  ents = [] # entities
  asps = [] # aspects
  labels = [] # polarity labels
  for sent, ops in train_data.items():
    # remove punctuations
    vocab.extend([word.lower() for word in sent.strip().split(' ') if word not in string.punctuation])
    for op in ops:
      cats = op['category'].lower().split('#')
      ents.append(cats[0])
      asps.append(cats[1])
      labels.append(op['polarity']) 
  for sent, ops in dev_data.items():
    for op in ops:
      cats = op['category'].lower().split('#')
      ents.append(cats[0])
      asps.append(cats[1])
      labels.append(op['polarity']) 
  for sent, ops in test_data.items():
    for op in ops:
      cats = op['category'].lower().split('#')
      ents.append(cats[0])
      asps.append(cats[1])
      labels.append(op['polarity']) 

  vocab.append('unk')
  vocab = set(vocab)
  word2idx = dict(zip(vocab, range(len(vocab))))
  idx2word = dict(zip(range(len(vocab)), vocab))

  ents = set(ents)
  ent2idx = dict(zip(ents, range(len(ents))))
  idx2ent = dict(zip(range(len(ents)), ents))

  asps = set(asps)
  asp2idx = dict(zip(asps, range(len(asps))))
  idx2asp = dict(zip(range(len(asps)), asps))

  labels = set(labels)
  label2idx = dict(zip(labels, range(len(labels))))
  idx2label = dict(zip(range(len(labels)), labels))

  return word2idx, idx2word, ent2idx, idx2ent, asp2idx, idx2asp, label2idx, idx2label


def prepSeq(seq, to_idx, cuda):
  idxs = [to_idx[w] if w in to_idx else to_idx['unk'] for w in seq]
  tensor = torch.LongTensor(idxs)
  return autograd.Variable(tensor.cuda()) if cuda else autograd.Variable(tensor)


def prepSeqs(data, word2idx, ent2idx, asp2idx, lb2idx, cuda):
  new_data = []
  for sent, ops in data.items():
    tok_seq = prepSeq([w.lower() for w in sent.strip().split(' ')], word2idx, cuda)
    ents = []
    asps = []
    lbs = []
    for op in ops:
      cats = op['category'].lower().split('#')
      ents.append(cats[0])
      asps.append(cats[1])
      lbs.append(op['polarity'])
    ent_seq = prepSeq(ents, ent2idx, cuda)
    asp_seq = prepSeq(asps, asp2idx, cuda)
    lb_seq = prepSeq(lbs, lb2idx, cuda)
    new_data.append([tok_seq, ent_seq, asp_seq, lb_seq])
  return new_data


def train(train_seqs, analyzer, criterion, optimizer, cuda):
  # clear grad
  optimizer.zero_grad()
  # training mode
  analyzer.train()
  total_loss = torch.cuda.FloatTensor([0]) if cuda else torch.FloatTensor([0])
  acc = 0
  # number of instances
  n_inst = 0
  for train_seq, ent_seq, asp_seq, lb_seq in train_seqs:
    probs = analyzer(train_seq, ent_seq, asp_seq)
    loss = criterion(probs, lb_seq)
    acc += (probs.max(1)[1] == lb_seq).data[0]
    total_loss += loss.data
    n_inst += len(lb_seq) 
    loss.backward()
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  # update minibatch
  optimizer.step()
  return total_loss, acc


def test(seqs, analyzer, criterion, cuda):
  analyzer.eval()
  total_loss = torch.cuda.FloatTensor([0]) if cuda else torch.FloatTensor([0])
  acc = 0
  n_inst = 0
  for seq, ent_seq, asp_seq, lb_seq in seqs:
    probs = analyzer(seq, ent_seq, asp_seq)
    loss = criterion(probs, lb_seq)
    acc += (probs.max(1)[1] == lb_seq).data[0]
    total_loss += loss.data
    n_inst += len(lb_seq) 
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  return total_loss, acc
  
