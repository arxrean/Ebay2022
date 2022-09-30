import os
import pdb
from string import punctuation
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
import random
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import DebertaV2TokenizerFast
from transformers import RobertaTokenizerFast

import pytorch_lightning as pl
import torch.nn.functional as F

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
			self.tokenizer = DebertaV2TokenizerFast.from_pretrained(opt.backbone)
		elif 'robert' in opt.backbone:
			self.tokenizer = RobertaTokenizerFast.from_pretrained(opt.backbone)
		else:
			self.tokenizer = BertTokenizerFast.from_pretrained(opt.backbone)

		self.tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}
		self.tag2token = defaultdict(set)
		for text, tags in zip(self.texts, self.tags):
			seg_text, seg_tags = self.get_seg(text, tags)
			for t, g in zip(seg_text, seg_tags):
				self.tag2token[g].add(t)
		total_tokens = sum(len(x) for x in self.tag2token.values())
		self.tag2distri = {k:len(v)/total_tokens for k, v in self.tag2token.items()}

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = self.texts[item]
		tags = self.tags[item]

		# replace_token_with_same_tag
		if self.mode == 'train':
			if self.opt.replace_token_with_same_tag > 0:
				seg_text, seg_tags = self.get_seg(text, tags)
				replace_token_with_same_tag = np.random.binomial(1, self.opt.replace_token_with_same_tag, len(seg_text))
				for i in range(len(seg_text)):
					if replace_token_with_same_tag[i]:
						seg_text[i] = np.random.choice(list(self.tag2token[seg_tags[i]]))
				_text, _tags = [], []
				for i in range(len(seg_text)):
					_text.append(seg_text[i])
					_tags.append([seg_tags[i]]+len(seg_text[i].split()[1:])*['UNK'])
				_text, _tags = ' '.join(_text).split(), np.concatenate(_tags)
				assert len(_text) == len(_tags)
				text, tags = _text, _tags

			if self.opt.shuffle_token_with_same_tag > 0:
				seg_text, seg_tags = self.get_seg(text, tags)
				_seg_text, _seg_tags = [], []
				for i in range(len(seg_text)):
					if i > 0 and seg_tags[i] == seg_tags[i-1]:
						_seg_text[-1] += seg_text[i].split()
						_seg_tags[-1] += [(seg_tags[i], len(seg_text[i].split()))]
					else:
						_seg_text.append(seg_text[i].split())
						_seg_tags.append([(seg_tags[i], len(seg_text[i].split()))])
				seg_text, seg_tags = _seg_text, _seg_tags
				for i in range(len(seg_text)):
					if len(seg_text[i]) > 1:
						seg_text[i] = list(np.random.permutation(seg_text[i]))
				_text, _tags = [], []
				for seg_t, seg_g in zip(seg_text, seg_tags):
					for g in seg_g:
						_text.append(' '.join(seg_t[:g[1]]))
						_tags.append(g[0])
						seg_t = seg_t[g[1]:]
				seg_text, seg_tags = _text, _tags
				_text, _tags = [], []
				for i in range(len(seg_text)):
					_text.append(seg_text[i])
					_tags.append([seg_tags[i]]+len(seg_text[i].split()[1:])*['UNK'])
				_text, _tags = ' '.join(_text).split(), np.concatenate(_tags)
				assert len(_text) == len(_tags)
				text, tags = _text, _tags
			
			if self.opt.drop_mention > 0:
				seg_text, seg_tags = self.get_seg(text, tags)
				drop_mentions = np.random.binomial(1, self.opt.drop_mention, len(seg_text))
				while drop_mentions.sum() == 0:
					drop_mentions = np.random.binomial(1, self.opt.drop_mention, len(seg_text))
				_text, _tags = [], []
				for i in range(len(seg_text)):
					if drop_mentions[i]:
						_text.append(seg_text[i])
						_tags.append([seg_tags[i]]+len(seg_text[i].split()[1:])*['UNK'])
				_text, _tags = ' '.join(_text).split(), np.concatenate(_tags)
				assert len(_text) == len(_tags)
				text, tags = _text, _tags

		tokens, labels = self.align_label(text, tags)

		return tokens['input_ids'][0], tokens['attention_mask'][0], torch.tensor(labels, dtype=torch.long)
	

	def get_seg(self, text, tags):
		seg_text, seg_tags = [], []
		idx = 0
		while idx < len(text):
			if tags[idx] != 'UNK':
				seg_text.append(text[idx])
				seg_tags.append(tags[idx])
			else:
				seg_text[-1] += ' ' + text[idx]
			idx += 1
		
		return seg_text, seg_tags


	def align_label(self, texts, labels):
		tokenized_inputs = self.tokenizer(' '.join(texts), padding='max_length', max_length=self.opt.max_len, truncation=True, return_tensors="pt")
		terms = self.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])
		# terms[1] = '{}{}'.format(chr(288), terms[1])

		label_idx = 0
		label_ids = []
		for i, t in enumerate(terms):
			if t in self.get_stop_tokens():
				label_ids.append(0)
			else:
				label_ids.append(self.tag_mapping.get(labels[label_idx], 0))
				if i < len(terms)-1 and terms[i+1][0] == chr(9601):
					label_idx += 1

		return tokenized_inputs, label_ids


	def get_stop_tokens(self):
		if 'deberta' in self.opt.backbone:
			return ['[CLS]', '[SEP]', '[PAD]']
		elif 'robert' in self.opt.backbone:
			return ['<s>', '</s>', '<pad>']
		else:
			return ['[CLS]', '[SEP]', '[PAD]']


	def get_split_token(self):
		if 'deberta' in self.opt.backbone:
			return chr(288)
		elif 'robert' in self.opt.backbone:
			return chr(288)
		else:
			return '##'


class QuizDataset:
	def __init__(self, opt, df):
		# input is annotated data frame
		self.df = df
		self.opt = opt

		self.ids = df['Record Number'].to_list()
		self.texts = [[preprocess_token(x) for x in ls] for ls in df['Title'].str.split(' ').to_list()]
		print(f"- max tokens: {max([len(x) for x in self.texts])}, min tokens: {min([len(x) for x in self.texts])}, avg tokens: {np.mean([len(x) for x in self.texts])}")
		
		if 'deberta' in opt.backbone:
			self.tokenizer = DebertaV2TokenizerFast.from_pretrained(opt.backbone)
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

		return tokens['input_ids'][0], tokens['attention_mask'][0], torch.tensor(labels, dtype=torch.long), self.ids[item]


	def align_label(self, texts, labels):
		tokenized_inputs = self.tokenizer(' '.join(texts), padding='max_length', max_length=self.opt.max_len, truncation=True, return_tensors="pt")
		terms = self.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])
		# terms[1] = '{}{}'.format(chr(288), terms[1])

		label_idx = 0
		label_ids = []
		for i, t in enumerate(terms):
			if t in self.get_stop_tokens():
				label_ids.append(0)
			else:
				label_ids.append(self.tag_mapping.get(labels[label_idx], 0))
				if i < len(terms)-1 and terms[i+1][0] == chr(9601):
					label_idx += 1

		return tokenized_inputs, label_ids

	def get_stop_tokens(self):
		if 'deberta' in self.opt.backbone:
			return ['[CLS]', '[SEP]', '[PAD]']
		elif 'robert' in self.opt.backbone:
			return ['<s>', '</s>', '<pad>']
		else:
			return ['[CLS]', '[SEP]', '[PAD]']


	def get_split_token(self):
		if 'deberta' in self.opt.backbone:
			return chr(288)
		elif 'robert' in self.opt.backbone:
			return chr(288)
		else:
			return '##'


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
