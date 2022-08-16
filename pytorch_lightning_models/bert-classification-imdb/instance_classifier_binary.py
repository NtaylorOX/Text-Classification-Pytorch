import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders import LabelEncoder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup 
from transformers.optimization import Adafactor, AdafactorSchedule 
from transformers import RobertaTokenizerFast as RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score 

from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.font_manager import FontProperties

from collections import defaultdict, namedtuple
from typing import *

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset


from typing import Union

import argparse

from loguru import logger
import warnings


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes

    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('normal')

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.95
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # figure.savefig(f'experiments/{model}/test_mtx.png')

    return figure

# data class
class InstanceDataset(Dataset):
    def __init__(self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len: int = 512, mode = "train"):

        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.mode = mode
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        all_text = data_row["review"]
        labels = data_row["label"]
        encoding = self.tokenizer.encode_plus(
          all_text,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        #TODO  - implement a balancing of the dataset - i.e. have around 50-50 split of labels

        return dict(
          all_text=all_text,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=torch.tensor(labels)
        )

# data module class - wrapped around pytorch lightning data module
class InstanceDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df,test_df, tokenizer, batch_size=2, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        logger.warning(f"size of training dataset: {len(train_df)} ")
        logger.warning(f"size of validation dataset: {len(valid_df)} ")
        logger.warning(f"size of test dataset: {len(test_df)} ")




    def setup(self, stage=None):
        self.train_dataset = InstanceDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.valid_dataset = InstanceDataset(
            self.valid_df,
            self.tokenizer,
            self.max_token_len
        )
        self.test_dataset = InstanceDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True

        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size

        )

# Bert model base

class InstanceModel(pl.LightningModule):
    def __init__(self,
                 model,
                 num_labels,
                 bert_hidden_dim=768,
                 classifier_hidden_dim=768,
                 n_training_steps=None,
                 n_warmup_steps=None,
                 dropout=0.2,
                 weight_classes=False,
                 #weights=torch.tensor([0.5, 1.5]),
                 class_labels = ['negative', 'positive'],
                 reinit_n_layers=2,
                 pretrained_dir ="F:/OxfordTempProjects/PatientTriageNLP/nlp_development/pretrained_hf_models/" ):

        super().__init__()
        logger.warning(f"Building model based on following architecture. {model}")
        self.num_labels = num_labels
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(f"{pretrained_dir}/{model}/model", return_dict=True)
        # nn.Identity does nothing if the dropout is set to None
        self.class_labels = class_labels
        self.classifier = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                        nn.Linear(classifier_hidden_dim, num_labels))
        #reinitialize n layers
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0:
            logger.warning(f"Re-initializing the last {reinit_n_layers} layers of encoder")
            self._do_reinit()
        #if we want to bias loss based on class sample sizes
        if weight_classes:
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

    def _do_reinit(self):
        # re-init pooler
        self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        self.model.pooler.dense.bias.data.zero_()
        for param in self.model.pooler.parameters():
            param.requires_grad = True

        # re-init last n layers
        for n in range(self.reinit_n_layers):
            self.model.encoder.layer[-(n + 1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        # obtaining the last layer hidden states of the Transformer
        last_hidden_state = output.last_hidden_state  # shape: (batch_size, seq_length, bert_hidden_dim)

        #         or can use the output pooler : output = self.classifier(output.pooler_output)
        # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation
        # by indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        logits = self.classifier(CLS_token_state)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs.detach(), "labels": labels.detach()}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs.detach(), "labels": labels.detach()}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs.detach(), "labels": labels.detach()}


    def validation_epoch_end(self, outputs):
        logger.warning("on validation epoch end")

        # get class labels
        class_labels = self.class_labels


        labels = []
        predictions = []
        scores = []
        for output in outputs:
            
            for out_labels in output["labels"].to('cpu').detach().numpy():                                
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                
                # the handling of roc_auc score differs for binary and multi class
                if len(class_labels) > 2:
                    scores.append(torch.nn.functional.softmax(out_predictions).cpu().tolist())
                # append probas
                else:
                    scores.append(torch.nn.functional.softmax(out_predictions)[1].cpu().tolist())

                # get predictied labels                               
                predictions.append(np.argmax(out_predictions.to('cpu').detach().numpy(), axis = -1))

            #use softmax to normalize, as the sum of probs should be 1

        # get epoch loss
        batch_losses = [x["loss"]for x in outputs] #This part
        epoch_loss = torch.stack(batch_losses).mean() 

        logger.info(f"labels are: {labels}")
        logger.info(f"predictions are: {predictions}")
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average = 'weighted')
        f1_macro = f1_score(labels, predictions, average = 'macro')
        prec_weighted = precision_score(labels, predictions, average = 'weighted')
        prec_macro = precision_score(labels, predictions, average = 'macro')
        recall_weighted = recall_score(labels, predictions, average = 'weighted')
        recall_macro = recall_score(labels, predictions, average = 'macro')

        #  roc_auc  - only really good for binaryy classification but can try for multiclass too
        # pytorch lightning runs a sanity check and roc_score fails if not all classes appear...
        try:
            if len(class_labels) > 2:  
                roc_auc_weighted = roc_auc_score(labels, scores, average = "weighted", multi_class = "ovr")
                roc_auc_macro = roc_auc_score(labels, scores, average = "macro", multi_class = "ovr")         
            else:
                roc_auc_weighted = roc_auc_score(labels, scores, average = "weighted")
                roc_auc_macro = roc_auc_score(labels, scores, average = "macro") 
        except ValueError:              
            logger.warning("roc_scores not calculated due to value error - caused by not all classes present in batch")
            roc_auc_weighted = 0
            roc_auc_macro = 0

        print(f"scores are: {scores}")
        print(f"labels are: {labels}")
        print(f"predictions are: {predictions}")
        print(f"roc scores: {roc_auc_macro}")
        # get confusion matrix
        cm = confusion_matrix(labels,predictions)

        # make plot 
        cm_figure = plot_confusion_matrix(cm, class_labels)
        
        # log this for monitoring
        self.log('monitor_balanced_accuracy', acc)
        self.log('monitor_roc_auc', roc_auc_macro)

        logger.warning(f"current epoch : {self.current_epoch}")

        # log to tensorboard
        self.logger.experiment.add_figure("valid/confusion_matrix", cm_figure, self.current_epoch)
        self.logger.experiment.add_scalar('valid/balanced_accuracy',acc, self.current_epoch)
        self.logger.experiment.add_scalar('valid/prec_weighted',prec_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('valid/prec_macro',prec_macro, self.current_epoch)
        self.logger.experiment.add_scalar('valid/f1_weighted',f1_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('valid/f1_macro',f1_macro, self.current_epoch)
        self.logger.experiment.add_scalar('valid/recall_weighted',recall_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('valid/recall_macro',recall_macro, self.current_epoch)
        self.logger.experiment.add_scalar('valid/roc_auc_weighted',roc_auc_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('valid/roc_auc_macro',roc_auc_macro, self.current_epoch)
        self.logger.experiment.add_scalar('valid/loss',epoch_loss, self.current_epoch)

    def test_epoch_end(self, outputs):
        # get class labels
        class_labels = self.class_labels


        labels = []
        predictions = []
        scores = []
        for output in outputs:
            
            for out_labels in output["labels"].to('cpu').detach().numpy():                                
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                
                # the handling of roc_auc score differs for binary and multi class
                if len(class_labels) > 2:
                    scores.append(torch.nn.functional.softmax(out_predictions).cpu().tolist())
                # append probas
                else:
                    scores.append(torch.nn.functional.softmax(out_predictions)[1].cpu().tolist())

                # get predictied labels                               
                predictions.append(np.argmax(out_predictions.to('cpu').detach().numpy(), axis = -1))

            #use softmax to normalize, as the sum of probs should be 1
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average = 'weighted')
        f1_macro = f1_score(labels, predictions, average = 'macro')
        prec_weighted = precision_score(labels, predictions, average = 'weighted')
        prec_macro = precision_score(labels, predictions, average = 'macro')
        recall_weighted = recall_score(labels, predictions, average = 'weighted')
        recall_macro = recall_score(labels, predictions, average = 'macro')

        #  roc_auc  - only really good for binaryy classification but can try for multiclass too
        if len(class_labels) > 2:   
            roc_auc_weighted = roc_auc_score(labels, scores, average = "weighted", multi_class = "ovr")
            roc_auc_macro = roc_auc_score(labels, scores, average = "macro", multi_class = "ovr")         
        else:
            roc_auc_weighted = roc_auc_score(labels, scores, average = "weighted")
            roc_auc_macro = roc_auc_score(labels, scores, average = "macro") 
 
        # get confusion matrix
        cm = confusion_matrix(labels,predictions)

        # make plot 
        cm_figure = plot_confusion_matrix(cm, class_labels)
        
        # log this for monitoring
        self.log('monitor_balanced_accuracy', acc)

        logger.warning(f"current epoch : {self.current_epoch}")

        # log to tensorboard
        self.logger.experiment.add_figure("test/confusion_matrix", cm_figure, self.current_epoch)
        self.logger.experiment.add_scalar('test/balanced_accuracy',acc, self.current_epoch)
        self.logger.experiment.add_scalar('test/prec_weighted',prec_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('test/prec_macro',prec_macro, self.current_epoch)
        self.logger.experiment.add_scalar('test/f1_weighted',f1_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('test/f1_macro',f1_macro, self.current_epoch)
        self.logger.experiment.add_scalar('test/recall_weighted',recall_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('test/recall_macro',recall_macro, self.current_epoch)
        self.logger.experiment.add_scalar('test/roc_auc_weighted',roc_auc_weighted, self.current_epoch)
        self.logger.experiment.add_scalar('test/roc_auc_macro',roc_auc_macro, self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=1e-3)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            #num_warmup_steps=0,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        logger.warning(f"Optimizer set up with the following parameters: {optimizer}")
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

