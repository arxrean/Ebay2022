import os
import re
import pdb
from tqdm import tqdm
import argparse
import string
import random
import pandas as pd
import numpy as np
import wandb

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import QuizDataset
from model import BertBaseline

from baseline import get_parser


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

	csv = pd.read_csv(opt.quiz_path)
	dataset = QuizDataset(opt, pd.read_csv(opt.quiz_path))
	dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

	model = BertBaseline(opt)
	model.eval()
	model = model.cuda()

	wandb_logger = WandbLogger()
	artifact = wandb_logger.experiment.use_artifact(opt.ckpt, type='model')
	artifact_dir = artifact.download()
	model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.ckpt'))['state_dict'])

	submission = []
	with torch.no_grad():
		for idx, pack in enumerate(tqdm(dataset_loader)):
			outs, input_ids, ids = model.forward(pack)
			label_indices = np.argmax(outs.cpu().numpy(), axis=2)

			for tokens, labels, id in zip(input_ids, label_indices, ids):
				new_tokens, new_labels = [], []
				for token, label_idx in zip(tokens, labels):
					if token in ['[CLS]', '[SEP]', '[PAD]']:
						continue
					if token.startswith("##"):
						new_tokens[-1] = new_tokens[-1] + token[2:]
					else:
						new_labels.append(label_idx)
						new_tokens.append(token)

				# print(csv[csv['Record Number']==id.item()]['Title'].tolist()[0])
				# if 'Black Satin Pleated Evening Handbag Wedding' in csv[csv['Record Number']==id.item()]['Title'].tolist()[0]:
				# 	pdb.set_trace()
				origin_text = csv[csv['Record Number']==id.item()]['Title'].tolist()[0].split()
				# if len(origin_text) > len(new_tokens):
				# 	print('origin: {} new: {}'.format(len(origin_text), len(new_tokens)))
				idx = 0
				while idx < len(new_tokens):
					if len(re.findall(r'[\u4e00-\u9fff]+', origin_text[idx])) > 0:
						print('skip chinese sentence')
						break
					if new_tokens[idx] == origin_text[idx] or new_tokens[idx]=='[UNK]':
						submission.append([id.item(), idx2tag.get(new_labels[idx], random.choice(list(tag_mapping.keys()))), origin_text[idx]])
						idx += 1
					else:
						s = new_tokens[idx]
						label_list = [new_labels[idx]]
						while s != origin_text[idx]:
							s += new_tokens[idx+1]
							label_list.append(new_labels[idx+1])
							new_tokens = new_tokens[:idx+1] + new_tokens[idx+2:]
							new_labels = new_labels[:idx+1] + new_labels[idx+2:]
						idx += 1
						majority = find_majority(label_list)[0]
						submission.append([id.item(), idx2tag.get(majority, random.choice(list(tag_mapping.keys()))), s])

	submission = pd.DataFrame(submission, columns=['Record Number', 'Aspect Name', 'Aspect Value'])
	submission = submission[submission['Aspect Name']!='No Tag']
	submission.to_csv('./dataset/ebay/submission.csv', index=False, header=False, sep='\t')
