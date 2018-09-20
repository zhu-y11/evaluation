# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Evaluation Metrics
@Author Yi Zhu
Upated 16.11.2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import scipy.stats as spst


def pearson(predictions, labels):
  return spst.pearsonr(predictions.numpy(), labels.numpy())


def spearmanr(predictions, labels):
  return spst.spearmanr(predictions.numpy(), labels.numpy())[0]
