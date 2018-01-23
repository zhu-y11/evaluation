# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Configurations for the main scripts
@Author Yi Zhu
Upated 01/23/2018
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
    #default = ['fasttext', 'he'],
    #default = ['glove', 'he'],
    #default = ['fasttext', 'en', 'de', 'it', 'ru'], 
    #default = ['polyglot', 'en', 'de', 'it', 'ru'], 
    #default = ['numberbatch', 'en', 'de', 'it', 'ru'], 
                        help = 'Embedding Model Name')
    parser.add_argument('--embedding_dir_path', default = '/media/hdd/yz568/data/word2vec',
                        help = 'Path to Embedding File')
    parser.add_argument('--embedding_file_name', nargs = '+',  
    default = ['en.wiki.word2vec'],
    #default = ['he.wiki.fasttext'],
    #default = ['he.wiki.glove'],
    #default = ['wiki.en.vec', 'wiki.de.vec', 'wiki.it.vec', 'wiki.es.vec', 'wiki.fa.vec'],
    #default = ['polyglot-en','polyglot-de', 'polyglot-it', 'polyglot-ru'],
    #default = ['polyglot-en','polyglot-de', 'polyglot-it', 'polyglot-es', 'polyglot-fa'],
                        help = 'Embedding File Name without Suffix')
    parser.add_argument('--lower_case', action = 'store_true', default = True)
    parser.add_argument('--cuda', action = 'store_true', default = True)
    # Evaluation Task
    parser.add_argument('--task', default = '_',
                        choices = ['word_similarity', 'pos_tagging', '_'],
                        help = 'Task Type')
    # Evaluation Data
    parser.add_argument('--evaldata_path', default = '/media/hdd/yz568/data/semeval16t5/subtask1/restaurant',
			help = 'Data Path for Evaluation Data')
    parser.add_argument('--evaldata_name', default = 'semeval16t5', 
    choices = ['simlex999', 'multi_simlex999', 
               'simverb3500', 'men3000', 'rareword', 'semeval17t2', 'semeval16t5'
               'wordsim353', 'wordsim353_rel', 'wordsim353_sim', 'multi_wordsim353', 'multi_wordsim353_rel', 'multi_wordsim353_sim',
               'UD_English', 'UD_Italian', 'UD_German', 'UD_Russian', 'UD_Finnish', 'UD_Turkish', 'UD_Hebrew', 'UD_Arabic',
	       'UD_Chinese', 'UD_Vietnamese'], 
                        help = 'Data Name')
    args = parser.parse_args()
    return args
