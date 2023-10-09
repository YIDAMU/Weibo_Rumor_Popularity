import torch
import numpy as np
from transformers import BertTokenizer
import ast
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.labels = [label for label in data['labels']]
        self.texts = [tokenizer(text, padding='max_length', max_length = 256, truncation=True,
                               return_tensors="pt") for text in data['text'].values]
        self.description = [tokenizer(des, padding='max_length', max_length = 64, truncation=True,
                               return_tensors="pt") for des in data['des'].values]
        self.profile = [profile.astype(np.float32) for profile in data[data.columns[:12]].values]
        self.kgss = [kgs.astype(np.float32) for kgs in data[data.columns[15:]].values]
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    def get_batch_texts(self, idx):
        return self.texts[idx]    
    def get_batch_description(self, idx):
        return self.description[idx]
    def get_batch_profile(self, idx):
        return self.profile[idx]
    def get_batch_kg(self, idx):
        return self.kgss[idx]    
    def __len__(self):
        return len(self.labels)
    def classes(self):
        return self.labels
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_description = self.get_batch_description(idx)
        batch_proflie=self.get_batch_profile(idx)
        batch_y = self.get_batch_labels(idx)
        batch_kg = self.get_batch_kg(idx)
        return batch_texts, batch_description, batch_proflie, batch_kg, batch_y 


