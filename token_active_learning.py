# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:41:21 2021

@author: Shadow
"""


import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.callbacks import EarlyStopping
import os 
import itertools
import pickle

from tokenize_data import *
from active_learner import *


val_data = get_single_ner('memc')
test_data = get_single_ner('memc', test = True)        

val_data, unique_labels = process_data(val_data, return_unique=True)
test_data = process_data(test_data)
    
data_directory = 'results/memc/token_inputs.pkl'      

with open(data_directory, 'rb') as f:
    train_data = pickle.load(f)

    
encoder_name = 'bert-base-uncased'

train_dataset = Token_Level_Dataset(input_ids = train_data['input_ids'], 
                                    attention_mask = train_data['attention_mask'], 
                                    token_idxs = train_data['token_idxs'],
                                    token_label_masks= train_data['token_label_masks'], 
                                    labels=train_data['train_labels'])


tokenizer = NER_tokenizer(unique_labels, max_length=64, tokenizer_name = encoder_name)

val_dataset = tokenizer.tokenize_and_encode_labels(val_data)
test_dataset = tokenizer.tokenize_and_encode_labels(test_data)

model = NER_ACTIVE_LEARNER(num_classes = len(tokenizer.id2tag), 
                 id2tag = tokenizer.id2tag,
                 tag2id = tokenizer.tag2id,
                 hidden_dropout_prob=.1,
                 attention_probs_dropout_prob=.1,
                 encoder_name = encoder_name,
                 save_fp='bert_token_memc.pt')

BATCH_SIZE = 64*32

model = train_LitModel(model, train_dataset, val_dataset, max_epochs=10, batch_size=BATCH_SIZE, patience = 3, num_gpu=1)