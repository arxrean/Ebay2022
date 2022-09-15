import os
import pdb
import argparse
import string
import random
import pandas as pd
import numpy as np
import wandb

from sklearn.model_selection import train_test_split

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import DERDatasetModule
from model import BertBaseline

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default=''.join(random.choice(string.ascii_lowercase) for i in range(10)))
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--accelerator', type=str, default='cpu')
	parser.add_argument('--test_only', action='store_true')

	parser.add_argument('--ner_path', type=str, default='./dataset/ebay/train_gr.csv')
	parser.add_argument('--quiz_path', type=str, default='./dataset/ebay/quiz.csv')
	parser.add_argument('--tag2idx', type=str, default='./dataset/ebay/tag2idx.npy')

	# data
	parser.add_argument('--train_frac', type=float, default=1.0)
	parser.add_argument('--val_frac', type=float, default=1.0)
	parser.add_argument('--test_frac', type=float, default=1.0)
	parser.add_argument('--train_val_split', type=float, default=0.3)
	parser.add_argument('--train_test_split', type=float, default=0.3)
	parser.add_argument('--val_pairs', type=int, default=1000)
	parser.add_argument('--test_pairs', type=int, default=1000)
	parser.add_argument('--classnum', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--max_len', type=int, default=64)
	parser.add_argument('--max_vocab', type=int, default=1000)
	parser.add_argument('--img_size', type=int, default=256)
	parser.add_argument('--crop_size', type=int, default=224)

	# train
	parser.add_argument('--log', action='store_true')
	parser.add_argument('--epoches', type=int, default=20)
	parser.add_argument('--base_lr', type=float, default=1e-5)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--warm_up_split', type=int, default=5)

	# model
	parser.add_argument('--pretrain', action='store_true')
	parser.add_argument('--backbone', type=str, default='dslim/bert-base-NER')
	parser.add_argument('--embed_dim', type=int, default=32)
	parser.add_argument('--hidden_dim', type=int, default=32)
	parser.add_argument('--word_embed_size', type=int, default=128)

	# test
	parser.add_argument('--ckpt', type=str, default='')

	opt = parser.parse_args()

	return opt


def main():
	opt = get_parser()
	seed_everything(42, workers=True)

	data_module = DERDatasetModule(opt)

	model = BertBaseline(opt)
	wandb_logger = WandbLogger(project="Ebay", log_model=True)
	wandb_logger.experiment.config.update(vars(opt))
	model.wandb_logger = wandb_logger

	checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
	trainer = Trainer(accelerator=opt.accelerator, devices=1, logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=opt.epoches)

	if not opt.test_only:
		trainer.fit(model, data_module)
		trainer.test(datamodule=data_module, ckpt_path="best")
		# trainer.test(datamodule=data_module, model=model)
	else:
		artifact = wandb_logger.experiment.use_artifact(opt.ckpt, type='model')
		artifact_dir = artifact.download()
		model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.ckpt'))['state_dict'])
		trainer.test(datamodule=data_module, model=model)


if __name__ == '__main__':
	main()