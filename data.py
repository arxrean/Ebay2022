import os
import pdb
import pandas as pd
import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import pytorch_lightning as pl

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader 



class NERDataset:
	def __init__(self, opt, df):
		# input is annotated data frame
		self.opt = opt
		self.texts = [eval(x) for x in df['Token'].to_list()]
		self.tags = [eval(x) for x in df['Tag'].to_list()]
		model_checkpoint = opt.backbone
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

		self.tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = self.texts[item]
		tags = self.tags[item]

		ids = []
		target_tag = []

		# tokenize words and define tags accordingly
		# running -> [run, ##ning]
		# tags - ['O', 'O']
		for i, s in enumerate(text):
			inputs = self.tokenizer.encode(s, add_special_tokens=False)
			input_len = len(inputs)
			ids.extend(inputs)
			target_tag.extend([self.tag_mapping[tags[i]]] * input_len)

		# truncate
		ids = ids[:self.opt.max_len - 2]
		target_tag = target_tag[:self.opt.max_len - 2]

		# add special tokens
		ids = [101] + ids + [102]
		target_tag = [0] + target_tag + [0]
		mask = [1] * len(ids)
		token_type_ids = [0] * len(ids)

		# construct padding
		padding_len = self.opt.max_len - len(ids)
		ids = ids + ([0] * padding_len)
		mask = mask + ([0] * padding_len)
		token_type_ids = token_type_ids + ([0] * padding_len)
		target_tag = target_tag + ([0] * padding_len)

		return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(target_tag, dtype=torch.long)


class QuizDataset:
	def __init__(self, opt, df):
		# input is annotated data frame
		self.df = df
		self.opt = opt
		self.ids = df['Record Number'].to_list()
		self.texts = df['Title'].str.split(' ').to_list()
		print(f"- max tokens: {max([len(x) for x in self.texts])}, min tokens: {min([len(x) for x in self.texts])}, avg tokens: {np.mean([len(x) for x in self.texts])}")
		model_checkpoint = opt.backbone
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

		self.tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = self.texts[item]

		ids = []
		target_tag = []

		# tokenize words and define tags accordingly
		# running -> [run, ##ning]
		# tags - ['O', 'O']
		for i, s in enumerate(text):
			inputs = self.tokenizer.encode(s, add_special_tokens=False)
			input_len = len(inputs)
			ids.extend(inputs)

		# truncate
		ids = ids[:self.opt.max_len - 2]

		# add special tokens
		ids = [101] + ids + [102]
		mask = [1] * len(ids)
		token_type_ids = [0] * len(ids)

		# construct padding
		padding_len = self.opt.max_len - len(ids)
		ids = ids + ([0] * padding_len)
		mask = mask + ([0] * padding_len)
		token_type_ids = token_type_ids + ([0] * padding_len)
		target_tag = [0] * self.opt.max_len

		return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(target_tag, dtype=torch.long), self.ids[item]


class DERDatasetModule(pl.LightningDataModule):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.df = pd.read_csv(opt.ner_path)

		self.data_train = NERDataset(opt, self.df[self.df['mode']==0])
		self.data_val = NERDataset(opt, self.df[self.df['mode']==1])
		self.data_test = NERDataset(opt, self.df[self.df['mode']==2])

	def train_dataloader(self):
		
		return DataLoader(self.data_train, batch_size = self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, drop_last=True)
  
	def val_dataloader(self):
		
		return DataLoader(self.data_val, batch_size = self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)
  
	def test_dataloader(self):
		
		return DataLoader(self.data_test, batch_size = self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)
