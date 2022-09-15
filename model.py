import os
import io
import pdb
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import AdamW, AutoModelForTokenClassification, get_scheduler

from sklearn.metrics import det_curve
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

from sklearn.metrics import f1_score, accuracy_score

import pytorch_lightning as pl

class BertBaseline(pl.LightningModule):
	def __init__(self, opt):
		super(BertBaseline, self).__init__()
		self.opt = opt
		self.wandb_logger = None
		self.tag_mapping = {k:v+1 for k,v in np.load(opt.tag2idx, allow_pickle=True).item().items()}

		model_checkpoint = opt.backbone
		self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(self.tag_mapping)+1, ignore_mismatched_sizes=True)
		# self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
		# self.model.classifier = nn.Linear(768, 1+len(self.tag_mapping))
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

	def forward(self, batch):
		### IMAGE #####
		input_ids, attention_mask, token_type_ids, labels, ids = batch
		input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()

		x = {'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels}
		outs = self.model(**x)

		return outs[1], [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids], ids

	def training_step(self, batch, batch_idx):
		input_ids, attention_mask, token_type_ids, labels = batch

		x = {'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels}
		outputs = self.model(**x)
		loss = outputs.loss

		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(input_ids))
		return loss

	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, token_type_ids, labels = batch

		x = {'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels}
		outputs = self.model(**x)
		loss = outputs.loss

		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		input_ids, attention_mask, token_type_ids, labels = batch

		x = {'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels}
		outputs = self.model(**x)

		return np.argmax(outputs[1].detach().cpu().numpy(), axis=2).ravel(), x['labels'].detach().cpu().numpy().ravel()

	def test_epoch_end(self, outputs):
		preds = np.concatenate([x[0] for x in outputs], 0)
		labels = np.concatenate([x[1] for x in outputs], 0)

		f1 = f1_score(preds, labels, average='macro')
		acc = accuracy_score(preds, labels)

		plt.bar(x=['f1', 'acc'], height=[
			f1, acc
		])
		self.wandb_logger.experiment.log({"test/evaluation": plt})
		plt.close()

	def configure_optimizers(self):
		param_optimizer = list(self.model.classifier.named_parameters())
		optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
		optimizer = AdamW(
			optimizer_grouped_parameters,
			lr=self.opt.base_lr,
			eps=1e-12
		)

		return optimizer




