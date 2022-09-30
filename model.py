import os
import io
import pdb
import wandb
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import AdamW, AutoModelForTokenClassification, get_scheduler
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
from transformers import RobertaTokenizerFast, RobertaForTokenClassification

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
		self.tag_mapping_inverse = dict((v, k) for k, v in self.tag_mapping.items())

		if 'deberta' in opt.backbone:
			# model = torch.load('./dataset/pytorch_model.bin.1')
			self.model = DebertaV2ForTokenClassification.from_pretrained(opt.backbone, num_labels=len(self.tag_mapping)+1, ignore_mismatched_sizes=True)
		elif 'robert' in opt.backbone:
			self.model = RobertaForTokenClassification.from_pretrained(opt.backbone, num_labels=len(self.tag_mapping)+1, ignore_mismatched_sizes=True)
		else:
			self.model = BertForTokenClassification.from_pretrained(opt.backbone, num_labels=len(self.tag_mapping)+1)

		if 'deberta' in opt.backbone:
			self.tokenizer = DebertaV2TokenizerFast.from_pretrained(opt.backbone)
		elif 'robert' in opt.backbone:
			self.tokenizer = RobertaTokenizerFast.from_pretrained(opt.backbone)
		else:
			self.tokenizer = BertTokenizerFast.from_pretrained(opt.backbone)

	def forward(self, batch):
		### IMAGE #####
		input_ids, attention_mask, labels, ids = batch
		if self.opt.accelerator == 'gpu':
			input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

		loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

		return logits, [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids], ids

	def training_step(self, batch, batch_idx):
		input_ids, attention_mask, labels = batch

		loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(logits))
		return loss

	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, labels = batch

		loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

		_logits, _labels = [], []
		for i in range(logits.shape[0]):
			logits_clean = logits[i][labels[i] != 0]
			label_clean = labels[i][labels[i] != 0]

			_logits.append(logits_clean)
			_labels.append(label_clean)

		_logits = torch.cat(_logits, 0)
		_labels = torch.cat(_labels, 0)

		predictions = _logits.argmax(dim=1)
		acc = (predictions == _labels).float().mean()
		self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		input_ids, attention_mask, labels = batch

		loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

		return logits.cpu().numpy(), labels.cpu().numpy()

	def test_epoch_end(self, outputs):
		preds = np.concatenate([x[0] for x in outputs], 0).argmax(-1)
		labels = np.concatenate([x[1] for x in outputs], 0)

		_preds = []
		_labels = []
		for pred, label in zip(preds, labels):
			_preds.append(pred[label!=0])
			_labels.append(label[label!=0])
		_preds = np.concatenate(_preds, 0)
		_labels = np.concatenate(_labels, 0)

		f1 = f1_score(_preds, _labels, average='macro')
		acc = accuracy_score(_preds, _labels)
		table = wandb.Table(data=[['f1', f1], ['acc', acc]], columns = ["metric", "value"])
		self.wandb_logger.experiment.log({"test/f1_acc" : wandb.plot.bar(table, "metric", "value", title="f1 and acc")})

		f1 = f1_score(_preds, _labels, average=None, labels=list(self.tag_mapping.values()))
		labels = [self.tag_mapping_inverse[x] for x in list(range(1, len(f1)+1))]
		table = wandb.Table(data=[[label, val] for (label, val) in zip(labels, f1)], columns = ["tag", "f1"])
		plt.bar(x=[self.tag_mapping_inverse[x] for x in list(range(1, len(f1)+1))], height=f1)
		self.wandb_logger.experiment.log({"test/f1_each" : wandb.plot.bar(table, "tag", "f1", title="f1 per tag")})

	def configure_optimizers(self):
		my_list = 'classifier'
		new_params = list(filter(lambda kv: my_list in kv[0], self.named_parameters()))
		base_params = list(filter(lambda kv: my_list not in kv[0], self.named_parameters()))

		params = [
                {'params': [x[1] for x in new_params]},
                {'params': [x[1] for x in base_params], 'lr': self.opt.base_lr/10}
        	]

		optimizer = AdamW(
			params,
			lr=self.opt.base_lr,
			eps=1e-12,
			weight_decay=self.opt.weight_decay
		)

		return optimizer




