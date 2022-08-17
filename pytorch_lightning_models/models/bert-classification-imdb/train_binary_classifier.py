import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loguru import logger


from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizerFast as RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from instance_classifier_binary import InstanceModel, InstanceDataset, InstanceDataModule
import argparse
from datetime import datetime
import warnings

import os 

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='8,9'

'''
Script to run training with a argument specified BERT model as the pre-trained encoder for instance classification.

We have a number of pretrained BERT models saved locally at: F:/OxfordTempProjects/PatientTriageNLP/nlp_development/pretrained_hf_models
Main task at the moment is a binary classification task - predicting a label of accept/reject based on a patients referral documents.

Example cmd usage:

python train_binary_classifier.py --data_dir F:/OxfordTempProjects/PatientTriageNLP/processed_data/instance_classification/balanced/concat/ --model emilyalsentzer/Bio_ClinicalBERT --accumulate_grad_batches 4
'''

# classes are imbalanced - lets calculate class weights for loss

def get_class_weights(train_df, label_col):
    classes = list(train_df[label_col].unique())
    class_dict = {}
    nSamples = []
    for c in classes:
        class_dict[c] = len(train_df[train_df[label_col] == c])
        nSamples.append(class_dict[c])

    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    return torch.FloatTensor(normedWeights)

def read_csv(data_dir,filename):
    return pd.read_csv(f"{data_dir}{filename}", index_col=None)

def main():
    parser = argparse.ArgumentParser()

    #TODO - add an argument to specify whether using balanced data then update directories based on that

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "/mnt/sdd/yizhang/binary_classification_bert/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--training_file",
                        default = "train.csv",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")
    parser.add_argument("--validation_file",
                        default = "valid.csv",
                        type=str,
                        help = "The default name of the training file")
    parser.add_argument("--test_file",
                        default = "test.csv",
                        type=str,
                        help = "The default name of hte test file")
    parser.add_argument("--pretrained_models_dir",
                        default="/mnt/sdd/yizhang/binary_classification_bert/",
                        type=str,
                        help="The data path to the directory containing local pretrained models from huggingface")
    parser.add_argument("--model",
                        default="roberta-base",
                        type=str,
                        help="bert encoder architecture - e.g. bert-base-uncased")

    parser.add_argument("--text_col",
                        default = "review",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default = "/mnt/sdd/yizhang/binary_classification_bert/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files"
                        )
    parser.add_argument("--num_classes",
                        default = 2,
                        type=int,
                        help = "number of classes for classification problem"
                        )

    parser.add_argument("--reinit_n_layers",
                        default = 2,
                        type=int,
                        help = "number of pretrained final bert encoder layers to reinitialize for stabilisation"
                        )
    parser.add_argument("--max_tokens",
                        default = 512,
                        type=int,
                        help = "Max tokens to be used in modelling"
                        )
    parser.add_argument("--num_epochs",
                        default = 10,
                        type=int,
                        help = "Number of epochs to train"
                        )
    parser.add_argument("--batch_size",
                        default = 8,
                        type=int,
                        help = "batch size for training"
                        )
    parser.add_argument("--accumulate_grad_batches",
                        default = 2,
                        type=int,
                        help = "number of batches to accumlate before optimization step"
                        )
    parser.add_argument("--balance_data",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")

    # TODO - add an argument to specify whether using balanced data then update directories based on that
    args = parser.parse_args()

    print(f"arguments provided are: {args}")
    # set up parameters
    data_dir = args.data_dir
    save_dir = args.save_dir
    logs_save_dir = f"lightning_logs/"
    ckpt_save_dir = f"checkpoints/"
    pretrained_dir = args.pretrained_models_dir
    pretrained_model_name = args.model
    max_tokens = args.max_tokens
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    n_labels = args.num_classes
    reinit_n_layers = args.reinit_n_layers
    accumulate_grad_batches = args.accumulate_grad_batches
    time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
    version = f"version_{time_now}"

    # if provided with balanced dataset alter the save directory
    if "balanced" in data_dir:
        print("Balanced in data path name - presuming this data is balanced so setting save directories accordingly!")
        version = f"balanced/{version}"
        ckpt_save_dir = ckpt_save_dir+"balanced"

        # automatically add balanced to save_dir regardless of whether balance function is applied here.
        save_dir = save_dir + "/balanced/"


    #TODO - same as above but for whether we have long or concat data
    seed_everything(42,workers = True)


    # load tokenizer
    print(f"loading tokenizer : {pretrained_dir}{pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_dir}/{pretrained_model_name}/tokenizer")

    # read in training and validation data
    train_df = read_csv(data_dir, args.training_file)
    val_df = read_csv(data_dir, args.validation_file)
    test_df = read_csv(data_dir, args.test_file)


    # push data through pipeline
    # instantiate datamodule
    data_module = InstanceDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=batch_size,
        max_token_len=max_tokens
    )

    steps_per_epoch = len(train_df) // batch_size
    total_training_steps = steps_per_epoch * n_epochs
    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    # get some class specific loss weights - only needed if doing some form of weighted cross entropy with ubalanced classes
    weights = get_class_weights(data_module.train_df, "label")

    #set up model
    model = InstanceModel(model=pretrained_model_name,
                                pretrained_dir = pretrained_dir,
    
                                 num_labels=n_labels,
                                 n_warmup_steps=warmup_steps,
                                 n_training_steps=total_training_steps,
                                # weights=weights,
                                 reinit_n_layers=reinit_n_layers
                                 )

    #setup checkpoint and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ckpt_save_dir}/{pretrained_model_name}/version_{time_now}",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last = True
    )

    pl_logger = TensorBoardLogger(save_dir="lightning_logs",
                               version=f"{version}",
                               name=f"{pretrained_model_name}")

    #trainer
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=4)
    trainer = pl.Trainer(
        logger=pl_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs = n_epochs,
        gpus = 2,
        accumulate_grad_batches= accumulate_grad_batches
    )

    # run training
    trainer.fit(model, data_module)

    # test
    # trainer.test(ckpt_path="best")

# run script
if __name__ == "__main__":
    main()