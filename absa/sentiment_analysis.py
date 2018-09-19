# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Multilingual Sentiment Analysis 
@Author Yi Zhu
Upated 06/06/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import random
from random import shuffle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)
random.seed(1234)

import logging
logger = logging.getLogger(__name__)

import pdb


class absaAnalyzer(nn.Module):
  def __init__(self, word2idx, idx2word, ent2idx, idx2ent, asp2idx, idx2asp, pol2idx, idx2pol, ea2idx, idx2ea, eapos2idx, idx2eapos, pos2idx, idx2pos, emb_vocab, emb, slt, max_seq_len, bs, hidden_dim = 750, bidirectional = True, num_layers = 2):
    super(absaAnalyzer, self).__init__()

    self.word2idx = word2idx
    self.idx2word = idx2word
    self.ent2idx = ent2idx
    self.idx2ent = idx2ent
    self.asp2idx = asp2idx
    self.idx2asp = idx2asp
    self.pol2idx = pol2idx
    self.idx2pol = idx2pol
    self.ea2idx = ea2idx
    self.idx2ea = idx2ea
    self.eapos2idx = eapos2idx
    self.idx2eapos = idx2eapos
    self.pos2idx = pos2idx
    self.idx2pos = idx2pos

    self.vocab_size = len(word2idx)
    self.vocab_pad = self.vocab_size
    self.ent_size = len(ent2idx)
    self.asp_size = len(asp2idx)
    self.pol_size = len(pol2idx)
    self.ea_size = len(ea2idx)
    self.eapos_size = len(eapos2idx)
    self.pos_size = len(pos2idx)
 
    self.max_seq_len = max_seq_len
    self.bs = bs

    # word embedding dim = pre-trained emb dim
    self.embed_dim = emb.size()[1]
    self.word_embeds = self.initEmbeds(emb_vocab, emb)    

    self.hidden_dim = hidden_dim
    self.bidirectional = bidirectional
    self.num_layers = num_layers

    logger.info('Initializing model for {}'.format(slt))
    if slt == 'slot1':
      # slot 1
      self.init_slt1()
    elif slt == 'slot2':
      # slot 2
      self.init_slt2()


  def initEmbeds(self, emb_vocab, emb):
    logger.info('initializing embedding layers ...')
    # initialize randomly with padding
    embeds = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx = self.vocab_size)
    for w, idx in tqdm(self.word2idx.items(), total = self.vocab_size):
      # skip words that are not in pre-trained embeddings and paddings
      if w not in emb_vocab:
        continue
      # initialize words that are in the pretrained embeddings
      embeds.weight.data[idx] = emb[emb_vocab.index(w)]
    return embeds


  def init_slt1(self):
    # conv layer
    self.filter_n = 300
    self.win = 5
    self.conv1 = nn.Conv2d(1, self.filter_n, (self.win, self.embed_dim))
    self.tanh = nn.Tanh()
    # polling
    self.maxpool = nn.MaxPool1d(self.max_seq_len - self.win + 1)
    #MLP
    self.h_dim = 100
    self.l1 = nn.Linear(self.filter_n, self.h_dim)
    self.l2 = nn.Linear(self.h_dim, self.ea_size)
    # threshold
    self.t = 0.2


  def init_slt2(self):
    self.dropout = nn.Dropout(p = 0.4)

    self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, 
                        num_layers = self.num_layers, dropout = 0.4, bidirectional = self.bidirectional)
    self.hidden = self.init_hidden()

    self.h2o = nn.Linear(self.hidden_dim, self.pos_size)    


  def init_hidden(self):
    bi_lstm = 2 if self.bidirectional else 1
    return (torch.randn(self.num_layers * bi_lstm, self.bs, self.hidden_dim // 2, requires_grad = True),
            torch.randn(self.num_layers * bi_lstm, self.bs, self.hidden_dim // 2, requires_grad = True))


  def forward(self, in_seqs): 
    sent_embeds = self.word_embeds(in_seqs).unsqueeze(1)

    conv_layer = self.tanh(self.conv1(sent_embeds).squeeze())

    pooling_layer = self.maxpool(conv_layer).squeeze()
    h1 = self.l1(pooling_layer)
    h2 = self.l2(h1)
    #probs = self.sigmoid(h2)
    return h2


    '''
    self.hidden = self.init_hidden()

    embeds = self.word_embeds(sent).view(len(sent), 1, -1)
    ent_embeds = self.ent_embeds(ents).view(len(ents), -1)
    asp_embeds = self.asp_embeds(asps).view(len(asps), -1)

    if self.training:
      embeds = self.dropout(embeds)
   
    # n = len(sent), m = len(ents)
    lstm_out, self.hidden = self.lstm(embeds) 
    # m, n, hidden_dim
    lstm_out = lstm_out.squeeze().unsqueeze(0).repeat(len(ents), 1, 1)
    # m, n, 2 * ea_dim
    ent_asp_embeds = torch.cat((ent_embeds, asp_embeds), 1).unsqueeze(1).repeat(1, len(sent), 1)
    # m, n, hidden_dim + 2 * ea_dim
    h_eas_cat = torch.cat((lstm_out, ent_asp_embeds), 2)

    # m, n, mlph_dim
    scr_h = F.relu(self.mlp_intoh(h_eas_cat))
    # m, n, 1
    scrs = self.mlp_htoo(scr_h)
    atts = F.softmax(scrs, 1)
    
    # m, n, hidden_dim
    lstm_out_att = lstm_out * atts
    # m, hidden_dim
    lstm_out_att = lstm_out_att.sum(1)
 
    os = self.atttoo(lstm_out_att)
    prob_os = F.log_softmax(os, 1)
    return prob_os
    '''


def evalEmbed(emb_vocab, emb, train_data, dev_data, test_data, lang, slt, lower_case, cuda, 
    bs = 100, max_seq_len = 60, epoch = 20, report_every = 5):
  if cuda:
    torch.cuda.manual_seed(1234)
  (word2idx, idx2word,
  ent2idx, idx2ent,
  asp2idx, idx2asp,
  pol2idx, idx2pol,
  ea2idx, idx2ea,
  eapos2idx, idx2eapos,
  pos2idx, idx2pos) = get_vocab_entity_aspect_polarity(train_data, dev_data, test_data, lower_case)
  analyzer = absaAnalyzer(word2idx, idx2word, 
                          ent2idx, idx2ent, 
                          asp2idx, idx2asp,
                          pol2idx, idx2pol, 
                          ea2idx, idx2ea,
                          eapos2idx, idx2eapos,
                          pos2idx, idx2pos,
                          emb_vocab, emb, slt, max_seq_len, bs)
  if cuda:
    analyzer.cuda()

  criterion1 = nn.BCEWithLogitsLoss()
  criterion2 = nn.NLLLoss()
  optimizer = optim.Adam(analyzer.parameters()) 

  ''' 
    [
      [
        Tensor([w1_idx, w2_idx, ... wn_idx, padding_idx, padding_idx, ...]),
        Tensor(slot1 target)
      ],
      ...
    ]
  '''
  #train_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, lower_case, max_seq_len, cuda)
  train_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, eapos2idx, pos2idx, slt, lower_case, max_seq_len, cuda)
  dev_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, eapos2idx, pos2idx, slt, lower_case, max_seq_len, cuda)
  test_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, eapos2idx, pos2idx, slt, lower_case, max_seq_len, cuda)

  best_r_dev = (.0, .0, .0)
  corbest_r_test = (.0, .0, .0)
  # total batch number
  n_batch = len(train_seqs) // bs if len(train_seqs) % bs == 0 else len(train_seqs) // bs + 1
  for i in range(epoch):
    # shuflle training data
    shuffle(train_seqs)
    for j in range(n_batch):
      batch_train = train_seqs[j * bs: (j + 1) * bs]
      total_loss_train, pre_train, rec_train, f_train = train(batch_train, analyzer, criterion, optimizer, cuda) 
      if j % report_every == 0:
        _, pre_dev, rec_dev, f_dev = test(train_seqs, analyzer, criterion, cuda)
        _, pre_test, rec_test, f_test = test(test_seqs, analyzer, criterion, cuda)
        print('current dev = {:.3f} {:.3f} {:.3f}, current test = {:.3f} {:.3f} {:.3f}'.format(pre_dev, rec_dev, f_dev, pre_test, rec_test, f_test))
        if f_dev > best_r_dev[-1]:
          best_r_dev = (pre_dev, rec_dev, f_dev)
          corbest_r_test = (pre_test, rec_test, f_test)
          #torch.save(analyzer, 'best_analyzer_{}.model'.format(lang))
      print("epoch {}, batch {}/{}\nloss = {:.5f}  train_f = {:.3f}\nbest dev_f = {:.3f}  corresponding test_f = {:.3f}".format(i + 1, j + 1, n_batch, total_loss_train / len(batch_train), f_train, best_r_dev[-1], corbest_r_test[-1]))


def get_vocab_entity_aspect_polarity(train_data, dev_data, test_data, lower_case):
  vocab = [] # vocabulary
  ents = [] # entities
  asps = [] # aspects
  pols = [] # polarity labels
  eas = [] # entity#aspect tuples
  eapos = ['O'] # entity#aspect_POS for slot 1&2
  pos = ['B', 'I', 'O'] # Position for slot 2

  get_all_terms(train_data, vocab, ents, asps, pols, eas, eapos, lower_case, is_train_data = True)
  get_all_terms(dev_data, vocab, ents, asps, pols, eas, eapos, lower_case)
  get_all_terms(test_data, vocab, ents, asps, pols, eas, eapos, lower_case)

  # append UNK to the last index
  vocab.append('UNK')
  vocab = set(vocab)
  word2idx = dict(zip(vocab, range(len(vocab))))
  idx2word = dict(zip(range(len(vocab)), vocab))

  ents = set(ents)
  ent2idx = dict(zip(ents, range(len(ents))))
  idx2ent = dict(zip(range(len(ents)), ents))

  asps = set(asps)
  asp2idx = dict(zip(asps, range(len(asps))))
  idx2asp = dict(zip(range(len(asps)), asps))

  pols = set(pols)
  pol2idx = dict(zip(pols, range(len(pols))))
  idx2pol = dict(zip(range(len(pols)), pols))

  eas = set(eas)
  ea2idx = dict(zip(eas, range(len(eas))))
  idx2ea = dict(zip(range(len(eas)), eas))

  eapos = set(eapos)
  eapos2idx = dict(zip(eapos, range(len(eapos))))
  idx2eapos = dict(zip(range(len(eapos)), eapos))

  pos2idx = dict(zip(pos, range(len(pos))))
  idx2pos = dict(zip(range(len(pos)), pos))

  return word2idx, idx2word, ent2idx, idx2ent, asp2idx, idx2asp, pol2idx, idx2pol, ea2idx, idx2ea, eapos2idx, idx2eapos, pos2idx, idx2pos


def get_all_terms(input_data, vocab, ents, asps, pols, eas, eapos, lower_case, is_train_data = False):
  for sent, ops in input_data.items():
    # only add words in trainin data to vocab
    if is_train_data:
      vocab.extend([word.lower() if lower_case else word for word in sent.strip().split(' ')])
    for op in ops:
      ea = op['category']
      cats = ea.split('#')
      eas.append(ea)
      eapos.extend([ea + '_B', ea + '_I'])
      ents.append(cats[0])
      asps.append(cats[1])
      pols.append(op['polarity']) 


def prepSeq(seq, to_idx, cuda, max_seq_len = -1):
  idxs = [to_idx[w] if w in to_idx else to_idx['UNK'] for w in seq] + [len(to_idx)] * (max_seq_len - len(seq))
  tensor = torch.LongTensor(idxs)
  return tensor.cuda() if cuda else tensor


def prepTarget(tokens, ops, ent2idx, asp2idx, pol2idx, ea2idx, eapos2idx, pos2idx, slt, cuda):
  sent = ' '.join(tokens)
  target = []
  pos = [pos2idx['O']] * len(tokens)
  for op in ops:
    #ent, asp = op['category'].split('#')
    if slt == 'slot1':
      ea = op['category']
      target.append(ea)
    elif slt == 'slot2':
      update_pos(sent, tokens, op, pos, pos2idx)

  if slt == 'slot1':
    target = list(set(target))
    target = prepSeq(target, ea2idx, cuda)
    target_vec = torch.zeros(len(ea2idx), dtype = torch.float)
    target_vec[target] = 1
    return target_vec
  elif slt == 'slot2':
    pos = torch.LongTensor(pos)
    if cuda:
      pos = pos.cuda()
    return pos


def update_pos(sent, tokens, op, pos, pos2idx):
  start = int(op['from'])
  end = int(op['to'])
  if start == 0 and end == 0:
    # NULL
    return
  else:
    start_idx = sent[:start].count(' ')
    end_idx = start_idx + 1 + sent[start: end].count(' ')
    assert(sent[start: end] == ' '.join(tokens[start_idx: end_idx]))
    pos[start_idx] = pos2idx['B']
    pos[start_idx + 1: end_idx] = [pos2idx['I']] * (end_idx - start_idx - 1)
    

def prepSeqs(data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, eapos2idx, pos2idx, slt, lower_case, max_seq_len, cuda):
  new_data = []
  for sent, ops in data.items():
    tokens = [w.lower() if lower_case else w for w in sent.strip().split(' ')]
    input_seq = prepSeq(tokens, word2idx, cuda, max_seq_len)
    target_seq = prepTarget(tokens, ops, ent2idx, asp2idx, pol2idx, ea2idx, eapos2idx, pos2idx, slt, cuda)
    pdb.set_trace()
    new_data.append([input_seq, target_seq])
  return new_data


def train(train_seqs, analyzer, criterion, optimizer, cuda):
  # clear grad
  optimizer.zero_grad()
  # training mode
  analyzer.train()
  # eval metrics
  cor = 0
  pred_n = 0
  real_n = 0
  in_seqs, target_seqs = zip(*train_seqs)
  in_seqs = torch.stack(in_seqs)
  target_seqs = torch.stack(target_seqs)
  real_n = target_seqs.sum()

  scores = analyzer(in_seqs)
  loss = criterion(scores, target_seqs)
  loss.backward()
  optimizer.step()

  sys_seqs = (F.sigmoid(scores) > analyzer.t).type(torch.float)
  pred_n = sys_seqs.sum()
  re_seqs = ((sys_seqs + target_seqs) == 2).type(torch.float)
  cor = re_seqs.sum()

  pre = cor / pred_n
  rec = cor / real_n
  f_score = 2 * pre * rec / (pre + rec)

  return  loss, pre, rec, f_score



def test(seqs, analyzer, criterion, cuda):
  # evaluation mode
  analyzer.eval()
  # eval metrics
  cor = 0
  pred_n = 0
  real_n = 0
  in_seqs, target_seqs = zip(*seqs)
  in_seqs = torch.stack(in_seqs)
  target_seqs = torch.stack(target_seqs)
  real_n = target_seqs.sum()

  scores = analyzer(in_seqs)
  sys_seqs = (F.sigmoid(scores) > analyzer.t).type(torch.float)
  pred_n = sys_seqs.sum()
  re_seqs = ((sys_seqs + target_seqs) == 2).type(torch.float)
  cor = re_seqs.sum()

  pre = cor / pred_n
  rec = cor / real_n
  f_score = 2 * pre * rec / (pre + rec)

  return None, pre, rec, f_score
  
