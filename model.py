import os
import io
import pdb
import clip
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
		self.ce_loss = nn.CrossEntropyLoss()

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
		
		self.clip = clip.load("ViT-B/32")[0]
		for param in self.clip.parameters():
			param.requires_grad = False
		self.classifier = nn.Sequential(
			nn.Dropout(0.1),
			nn.Linear(in_features=1024+512, out_features=len(self.tag_mapping)+1, bias=True)
		)

	def forward(self, batch):
		### IMAGE #####
		input_ids, attention_mask, labels, ids = batch
		if self.opt.accelerator == 'gpu':
			input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

		loss, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)


		return logits, [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids], ids
	
	def forward_train(self, batch):
		input_ids, attention_mask, labels, clip_tokens = batch
		_, _, hiddens = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False, output_hidden_states=True)

		text_features = self.clip.encode_text(clip_tokens.squeeze(1))
		text_features = text_features.unsqueeze(1).repeat(1, 64, 1)
		hiddens = torch.cat((hiddens[-1], text_features), -1)
		logits = self.classifier(hiddens)
		loss = self.ce_loss(logits.reshape(-1, 33), labels.reshape(-1))

		return loss, logits

	def training_step(self, batch, batch_idx):
		loss, logits = self.forward_train(batch)

		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(logits))
		return loss

	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, labels, clip_tokens = batch

		loss, logits = self.forward_train(batch)
		
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
		input_ids, attention_mask, labels, _ = batch

		loss, logits = self.forward_train(batch)

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
		params = [
                {'params': self.classifier.parameters()},
				{'params': self.clip.parameters(), 'lr': self.opt.base_lr/10},
                {'params': self.model.parameters(), 'lr': self.opt.base_lr/10}
        	]

		optimizer = AdamW(
			params,
			lr=self.opt.base_lr,
			eps=1e-12,
			weight_decay=self.opt.weight_decay
		)

		return optimizer




