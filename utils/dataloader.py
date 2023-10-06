# from cv2 import FlannBasedMatcher
import torch

from torchtext.data import Field, BucketIterator
from torchtext import data

import spacy
import numpy as np
import pandas as pd

import random
import os
import glob

# python -m spacy download en
# python -m spacy download de

# from preprocessing import preprocessing


def Multi30K(SRC, TRG, root = "data/multi30k", data_type = "train", train_data = "train.csv", test_data = "test.csv", val_data = "val.csv"):

    fields = [('src', SRC),('trg', TRG)]
    train_df = pd.read_csv(os.path.join(root, train_data))
    val_df = pd.read_csv(os.path.join(root, test_data))

    def get_dataset(df):
        example = [data.Example.fromlist([df.Ger[i],df.English[i]], fields) for i in range(df.shape[0])]
        dataset = data.Dataset(example, fields)
        return dataset
    if data_type == "train":
        return get_dataset(train_df)
    elif data_type == "val":
        return get_dataset(val_df)


def dataloaders(device, data_name, batch_size = None, seed=42):
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize = tokenize_de, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    if data_name == "multi30k":
        train_data = Multi30K(SRC, TRG, data_type = "train")
        # test_data = Multi30K(SRC, TRG, data_type = "test")
        valid_data = Multi30K(SRC, TRG, data_type = "val")

    print("\n****************************************************************************\n")
    print("*****Data Details*****\n")
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    # print(f"Number of testing examples: {len(test_data.examples)}")
    print("\n****************************************************************************\n")
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

    if cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    

    batch_size = batch_size or (128 if cuda else 64)

    train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    sort_key = lambda x: len(x.src),
    batch_size = batch_size, 
    device = device)

    return train_data, valid_data, train_iterator, valid_iterator, SRC, TRG
