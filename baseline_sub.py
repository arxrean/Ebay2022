import os
import re
import pdb
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

	csv = pd.read_csv(opt.quiz_path)
	dataset = QuizDataset(opt, pd.read_csv(opt.quiz_path))
	dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

	model = BertBaseline(opt)
	model.eval()
	model = model.cuda() if opt.accelerator == 'gpu' else model

	wandb_logger = WandbLogger()
	artifact = wandb_logger.experiment.use_artifact(opt.ckpt, type='model')
	artifact_dir = artifact.download()
	model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.ckpt'), map_location=torch.device('cpu'))['state_dict'])

	submission = []
	with torch.no_grad():
		for idx, pack in enumerate(dataset_loader):
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
