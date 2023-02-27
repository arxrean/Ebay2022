import os
import re
import pdb
import argparse
import string
import random
import pandas as pd
import numpy as np
import wandb
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import QuizDataset
from model import BertBaseline

from baseline import get_parser
from utils import preprocess_token


def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False


def find_majority(k):
	myMap = {}
	maximum = ( '', 0 ) # (occurring element, occurrences)
	for n in k:
		if n in myMap: myMap[n] += 1
		else: myMap[n] = 1

		# Keep track of maximum on the go
		if myMap[n] > maximum[1]: maximum = (n,myMap[n])

	return maximum


class MultiModel:
	def __init__(self, opt) -> None:
		self.opt = opt
		self.models = []
		self.weights = []
		for ckpt in opt.ckpt.split('~'):
			if isfloat(ckpt):
				self.weights.append(float(ckpt))
			else:
				model = BertBaseline(opt)
				model.eval()
				model = model.cuda() if opt.accelerator == 'gpu' else model
				model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['state_dict'])
				self.models.append(model)
		if not self.weights:
			self.weights = torch.tensor([1]*len(self.models)).cuda() if opt.accelerator == 'gpu' else torch.tensor([1]*len(self.models))
		else:
			self.weights = torch.tensor(self.weights).cuda() if opt.accelerator == 'gpu' else torch.tensor(self.weights)
		self.weights = F.softmax(self.weights).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
	
	def forward(self, pack):
		_outs, _input_ids, _ids = [], [], []
		for model in self.models:
			outs, input_ids, ids = model.forward(pack)
			_outs.append(outs)
			_input_ids.append(input_ids)
			_ids.append(ids)

		_outs = (torch.stack(_outs)*self.weights).sum(0)
		return _outs, _input_ids[0], _ids[0]


if __name__ == '__main__':
	opt = get_parser()
	seed_everything(42, workers=True)

	tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}
	idx2tag = {v:k for k, v in tag_mapping.items()}

	csv = pd.read_csv(opt.quiz_path)
	dataset = QuizDataset(opt, pd.read_csv(opt.quiz_path))
	dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

	multi_model = MultiModel(opt)

	submission = []
	with torch.no_grad():
		for idx, pack in enumerate(dataset_loader):
			outs, input_ids, ids = multi_model.forward(pack)
			label_indices = np.argmax(outs.cpu().numpy(), axis=2)
			for tokens, labels, id in zip(input_ids, label_indices, ids):
				new_tokens, new_labels = [], []
				for i, (token, label_idx) in enumerate(zip(tokens, labels)):
					if token in ['[CLS]', '[SEP]', '[PAD]']:
						continue
					if token.startswith(chr(9601)):
						new_labels.append(label_idx)
						new_tokens.append(token[1:])
					else:
						new_tokens[-1] = new_tokens[-1] + token


				origin_text = [x for x in csv[csv['Record Number']==id.item()]['Title'].tolist()[0].split()]
				idx = 0
				assert len(origin_text) == len(new_tokens)
				while idx < len(new_tokens):
					if new_labels[idx] != 0:
						submission.append([id.item(), idx2tag[new_labels[idx]], origin_text[idx]])
					else:
						last = submission.pop()
						last[-1] += ' {}'.format(origin_text[idx])
						submission.append(last)
					idx += 1

	submission = pd.DataFrame(submission, columns=['Record Number', 'Aspect Name', 'Aspect Value'])
	submission = submission[submission['Aspect Name']!='No Tag']
	submission.to_csv('./dataset/ebay/submission.csv', index=False, header=False, sep='\t')