# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Configurations for the main scripts
@Author Yi Zhu
Upated 22/11/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Evaluations for Word Embeddings')
    # Embeddings
    parser.add_argument('--embedding_model', nargs = '+', 
    default = ['word2vec', 'en'],
    #default = ['fasttext', 'en', 'de', 'it', 'ru'], 
    #default = ['polyglot', 'en', 'de', 'it', 'ru'], 
    #default = ['numberbatch', 'en', 'de', 'it', 'ru'], 
                        help = 'Embedding Model Name')
    parser.add_argument('--embedding_dir_path', default = '/Users/marce/Documents/Study/cam/code/word2vec-mac',
                        help = 'Path to Embedding File')
    parser.add_argument('--embedding_file_name', nargs = '+', 
    default = ['vectors'],
    #default = ['wiki.en.vec', 'wiki.de.vec', 'wiki.it.vec', 'wiki.es.vec', 'wiki.fa.vec'],
    #default = ['polyglot-en','polyglot-de', 'polyglot-it', 'polyglot-ru'],
    #default = ['polyglot-en','polyglot-de', 'polyglot-it', 'polyglot-es', 'polyglot-fa'],
                        help = 'Embedding File Name without Suffix')
    parser.add_argument('--lower_case', action = 'store_true', default = True)
    # Evaluation Data
    parser.add_argument('--evaldata_path', default = '/Users/marce/data/wordsim353/combined.tab',
                        help = 'Data Path for Evaluation Data')
    parser.add_argument('--evaldata_name', default = 'wordsim353', 
    choices = ['simlex999', 'multi_simlex999', 
               'simverb3500', 'men3000', 'rareword', 'semeval17t2', 
               'wordsim353', 'wordsim353_rel', 'wordsim353_sim', 'multi_wordsim353', 'multi_wordsim353_rel', 'multi_wordsim353_sim'],
                        help = 'Data Name')
    args = parser.parse_args()
    return args
