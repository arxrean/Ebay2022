import os
import re
import pdb
import argparse
import string
import random
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import QuizDataset
from model import BertBaseline

from baseline import get_parser
from utils import preprocess_token


def find_majority(k):
	myMap = {}
	maximum = ( '', 0 ) # (occurring element, occurrences)
	for n in k:
		if n in myMap: myMap[n] += 1
		else: myMap[n] = 1

		# Keep track of maximum on the go
		if myMap[n] > maximum[1]: maximum = (n,myMap[n])

	return maximum


if __name__ == '__main__':
	opt = get_parser()
	seed_everything(42, workers=True)

	tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}
	idx2tag = {v:k for k, v in tag_mapping.items()}
	addon = pd.read_csv('./dataset/ebay/addon.csv').sample(5000000)
	addon = addon.iloc[len(addon)//2:]
	addon = addon[addon['Title'].str.split().str.len() >= 3]

	dataset = QuizDataset(opt, addon)
	dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

	model = BertBaseline(opt)
	model.eval()
	model = model.cuda() if opt.accelerator == 'gpu' else model
	wandb_logger = WandbLogger()
	artifact = wandb_logger.experiment.use_artifact(opt.ckpt, type='model')
	artifact_dir = artifact.download()
	model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.ckpt'), map_location=torch.device('cpu'))['state_dict'])

	# draw score distri
	# fig_nums = []
	# with torch.no_grad():
	# 	for idx, pack in enumerate(tqdm(dataset_loader)):
	# 		outs, input_ids, ids = model.forward(pack)
	# 		input_ids = np.array(input_ids)
	# 		outs = F.softmax(outs, -1).cpu().numpy()
	# 		for out, input_id in zip(outs, input_ids):
	# 			out = np.array([out[i] for i in range(len(input_id)) if input_id[i] not in ['[CLS]', '[SEP]', '[PAD]']])
	# 			orders = np.flip(np.argsort(out, 1), 1)
	# 			fig_nums.append((out[range(len(out)), orders[:, 0]]/out[range(len(out)), orders[:, 1]]).min())

	# np.save('./repo/fig_nums.npy', fig_nums)
	# plt.boxplot(fig_nums)
	# plt.savefig('./repo/time_min_plot.png')
	# plt.close()

	submission, fig_nums = [], []
	with torch.no_grad():
		for all_idx, pack in enumerate(tqdm(dataset_loader)):
			outs, input_ids, ids = model.forward(pack)
			label_indices = np.argmax(outs.cpu().numpy(), axis=2)
			outs = F.softmax(outs, -1).cpu().numpy()

			for batch_idx, (tokens, labels, out, id) in enumerate(zip(input_ids, label_indices, outs, ids)):
				new_tokens, new_labels = [], []
				out = np.array([out[i] for i in range(len(tokens)) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']])
				orders = np.flip(np.argsort(out, 1), 1)
				# fig_nums.append((out[range(len(out)), orders[:, 0]]/out[range(len(out)), orders[:, 1]]).min())
				if (out[range(len(out)), orders[:, 0]]/out[range(len(out)), orders[:, 1]]).min() <= 10:
					continue
				for i, (token, label_idx) in enumerate(zip(tokens, labels)):
					if token in ['[CLS]', '[SEP]', '[PAD]']:
						continue
					if token.startswith(chr(9601)):
						new_labels.append(label_idx)
						new_tokens.append(token[1:])
					else:
						new_tokens[-1] = new_tokens[-1] + token

				origin_text = [x for x in addon[addon['Record Number']==id.item()]['Title'].tolist()[0].split()]
				idx = 0
				if len(origin_text) != len(new_tokens):
					print(' '.join(origin_text))
					continue
				_submission = []
				while idx < len(new_tokens):
					if new_labels[idx] != 0:
						_submission.append([id.item(), idx2tag[new_labels[idx]], origin_text[idx], '', 0])
					else:
						_submission.append([id.item(), 'UNK', origin_text[idx], '', 0])
						# last = submission.pop()
						# last[-1] += ' {}'.format(origin_text[idx])
						# submission.append(last)
					idx += 1
				if _submission[0][1] == 'UNK':
					continue
				submission.append([id.item(), [x[2] for x in _submission], [x[1] for x in _submission], '', 0])

	np.save('./repo/fig_nums.npy', fig_nums)
	submission = pd.DataFrame(submission, columns=['Record Number', 'Token', 'Tag', 'fill', 'mode'])
	submission.to_csv('./dataset/ebay/addon_train.csv', index=False)
