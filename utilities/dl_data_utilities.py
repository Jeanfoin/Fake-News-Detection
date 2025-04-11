from typing import (
    Tuple, 
    Optional, 
    Dict, 
    Any
)

import pandas as pd
import numpy as np
import re
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
import multiprocessing
import spacy
spacy.require_cpu()
nlp = spacy.load("en_core_web_sm")
import lightning as L


def anonymize_text(text):
    """
    Anonymizes sensitive information in a given text by replacing specific patterns and named entities with placeholders.

    Parameters:
    text (str): The input text to be anonymized.

    Returns:
    str: The anonymized text with sensitive information replaced by placeholders.
    """

    mention_pattern = r"@[\w\d_]+"
    hashtag_pattern = r"#[\w\d_]+"
    media_pattern = r"pic\.twitter\.com/\w+"
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    person_pattern = r'\b(trump|hillary|clinton|obama)\b'   
    
    doc = nlp(text)
    anonymized_text = text  # Start with the original text
    anonymized_text = re.sub(r'\s+', ' ', anonymized_text).strip()
    anonymized_text = re.sub(media_pattern, '[URL]', anonymized_text)
    anonymized_text = re.sub(url_pattern, '[URL]', anonymized_text)
    anonymized_text = re.sub(mention_pattern, '[TWI]', anonymized_text)
    anonymized_text = re.sub(hashtag_pattern, '[TWI]', anonymized_text)

    # Define a set of entity labels to anonymize
    entity_labels_to_anonymize = [
        "PERSON", "FAC", "ORG", "GPE", "LOC", "PRODUCT", 
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", 
        "PERCENT", "MONEY", "QUANTITY", "ORDINAL"
    ]
    
    # Loop through the recognized entities and replace them with placeholders
    for ent in doc.ents:
        if ent.label_ in entity_labels_to_anonymize:
            anonymized_text = anonymized_text.replace(ent.text, f"[{ent.label_}]")

    anonymized_text = re.sub(person_pattern, '[PERSON]', anonymized_text, flags=re.IGNORECASE)

    return anonymized_text

def anonymize_texts_parallel(texts, num_workers=4):
    """
    Anonymizes a list of texts in parallel using multiple workers.

    Parameters:
    texts (list of str): A list of input texts to be anonymized.
    num_workers (int, optional): The number of parallel workers to use. Default is 4.

    Returns:
    list of str: A list of anonymized texts with sensitive information replaced by placeholders.
    """
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(anonymize_text, texts)
    return results


class NewsTextDataset(Dataset):
    """
    A dataset class for handling news texts and their corresponding labels.

    Parameters:
    ids (np.ndarray): An array of IDs for the news texts.
    texts (np.ndarray): An array of news texts.
    labels (np.ndarray): An array of labels corresponding to the news texts.
    tokenizer: The tokenizer to be used for encoding the texts.
    max_token_len (int, optional): The maximum length of the tokenized texts. Default is 128.

    Attributes:
    tokenizer: The tokenizer used for encoding the texts.
    ids (np.ndarray): An array of IDs for the news texts.
    texts (np.ndarray): An array of news texts.
    labels (np.ndarray): An array of labels corresponding to the news texts.
    max_token_len (int): The maximum length of the tokenized texts.
    """
    def __init__(
        self,
        ids: np.ndarray,
        texts: np.ndarray,
        labels: np.ndarray,
        tokenizer,
        #device: torch.device,
        max_token_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.ids = ids
        self.texts = texts
        self.labels = labels
        self.max_token_len = max_token_len
        #self.device = device

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        ids = torch.tensor(self.ids[index], dtype=torch.long)
        news_text = self.texts[index]
        labels = torch.tensor(self.labels[index], dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            news_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "ids": ids,
            "news_text": news_text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


class NewsTextDataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling news text datasets.

    Parameters:
    train_dataset: The training dataset.
    val_dataset: The validation dataset.
    test_dataset: The test dataset.
    batch_size (int, optional): The batch size for the DataLoaders. Default is 32.
    num_workers (int, optional): The number of worker processes for the DataLoaders. Default is 2.
    pin_memory (bool, optional): Whether to pin memory for the DataLoaders. Default is True.

    Attributes:
    train_dataset: The training dataset.
    val_dataset: The validation dataset.
    test_dataset: The test dataset.
    batch_size (int): The batch size for the DataLoaders.
    num_workers (int): The number of worker processes for the DataLoaders.
    pin_memory (bool): Whether to pin memory for the DataLoaders.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            persistent_workers=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: The DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

def number_of_tokens(text, tokenizer):
    """
    Calculate the number of tokens in a given text using a specified tokenizer.

    Parameters:
    - text (str): The input text to be tokenized.
    - tokenizer: The tokenizer object used to encode the text into tokens.

    Returns:
    - int: The number of tokens in the input text.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(tokens)
