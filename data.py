import os
import pdb
from string import punctuation
import pandas as pd
import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import DebertaTokenizerFast
from transformers import RobertaTokenizerFast

import pytorch_lightning as pl

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader 

from utils import preprocess_token


class NERDataset:
	def __init__(self, opt, df, mode):
		# input is annotated data frame
		self.opt = opt
		self.mode = mode
		self.texts = [[preprocess_token(t) for t in eval(x)] for x in df['Token'].to_list()]
		self.tags = [eval(x) for x in df['Tag'].to_list()]
		for x, y in zip(self.texts, self.tags):
			assert len(x) == len(y)

		if 'deberta' in opt.backbone:
			self.tokenizer = DebertaTokenizerFast.from_pretrained(opt.backbone)
		elif 'robert' in opt.backbone:
			self.tokenizer = RobertaTokenizerFast.from_pretrained(opt.backbone)
		else:
			self.tokenizer = BertTokenizerFast.from_pretrained(opt.backbone)

		self.tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = self.texts[item]
		tags = self.tags[item]
		# print(' '.join(text))
		# if 'Pioneer Express Route' in ' '.join(text):
		# 	pdb.set_trace()

		tokens, labels = self.align_label(text, tags)

		return tokens['input_ids'][0], tokens['token_type_ids'][0], tokens['attention_mask'][0], torch.tensor(labels, dtype=torch.long)


	def align_label(self, texts, labels):
		tokenized_inputs = self.tokenizer(' '.join(texts), padding='max_length', max_length=self.opt.max_len, truncation=True, return_tensors="pt")

		word_ids = tokenized_inputs.word_ids()

		previous_word_idx = None
		label_ids = []

		for word_idx in word_ids:
			if word_idx is None:
				label_ids.append(0)
			elif word_idx != previous_word_idx:
				label_ids.append(self.tag_mapping.get(labels[word_idx], 0))
			else:
				label_ids.append(self.tag_mapping[labels[word_idx]] if self.opt.label_all_tokens else 0)
			previous_word_idx = word_idx

		return tokenized_inputs, label_ids


class QuizDataset:
	def __init__(self, opt, df):
		# input is annotated data frame
		self.df = df
		self.opt = opt

		self.ids = df['Record Number'].to_list()
		self.texts = [[preprocess_token(x) for x in ls] for ls in df['Title'].str.split(' ').to_list()]
		print(f"- max tokens: {max([len(x) for x in self.texts])}, min tokens: {min([len(x) for x in self.texts])}, avg tokens: {np.mean([len(x) for x in self.texts])}")
		
		if 'deberta' in opt.backbone:
			self.tokenizer = DebertaTokenizerFast.from_pretrained(opt.backbone)
		elif 'robert' in opt.backbone:
			self.tokenizer = RobertaTokenizerFast.from_pretrained(opt.backbone)
		else:
			self.tokenizer = BertTokenizerFast.from_pretrained(opt.backbone)

		self.tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = self.texts[item]
		tags = ['Brand'] * len(text)

		tokens, labels = self.align_label(text, tags)

		return tokens['input_ids'][0], tokens['token_type_ids'][0], tokens['attention_mask'][0], torch.tensor(labels, dtype=torch.long), self.ids[item]


	def align_label(self, texts, labels):
		tokenized_inputs = self.tokenizer(' '.join(texts), padding='max_length', max_length=self.opt.max_len, truncation=True, return_tensors="pt")

		word_ids = tokenized_inputs.word_ids()

		previous_word_idx = None
		label_ids = []

		for word_idx in word_ids:

			if word_idx is None:
				label_ids.append(0)

			elif word_idx != previous_word_idx:
				label_ids.append(self.tag_mapping[labels[word_idx]])
			else:
				label_ids.append(self.tag_mapping[labels[word_idx]] if self.opt.label_all_tokens else 0)
			previous_word_idx = word_idx

		return tokenized_inputs, label_ids


class DERDatasetModule(pl.LightningDataModule):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.df = pd.read_csv(opt.ner_path)

		self.data_train = NERDataset(opt, self.df[self.df['mode']==0], mode='train')
		self.data_val = NERDataset(opt, self.df[self.df['mode']==1], mode='val')
		self.data_test = NERDataset(opt, self.df[self.df['mode']==2], mode='test')

	def train_dataloader(self):
		
		return DataLoader(self.data_train, batch_size = self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, drop_last=True)
  
	def val_dataloader(self):
		
		return DataLoader(self.data_val, batch_size = self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)
  
	def test_dataloader(self):
		
		return DataLoader(self.data_test, batch_size = self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)
