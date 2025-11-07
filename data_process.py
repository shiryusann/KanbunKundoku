import torch
import json
from typing import List, Dict
from torch.utils.data import Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

def get_label_map(path:str):
    with open(path, "r", encoding = "utf-8") as file:
        labels = json.load(file)

    l2i = {"okurigana":labels[0], "particle":labels[1], "position":labels[2], "segmentation":labels[3], "partofspeech":labels[4],
           "dependencyarc":labels[5], "dependencytype":labels[6]}
    i2l = {i:{index:label for label, index in j.items()} for i, j in l2i.items()}

    return l2i, i2l

def pad_sentence(tensors:List[torch.tensor], max_len:int, i:int):
    padded_tensors = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        pad_tensor = torch.tensor([i] * pad_len, dtype = torch.long).to(t.device)
        padded_tensor = torch.cat([t, pad_tensor], dim = 0)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim = 0)

def kanbun_collate_fn(batch, pad_token_id):
    max_len = max([b[0]["input_ids"].size(0) for b in batch])
    
    pad_sentences = {}
    pad_labels = {}
    for l in batch[0][0]:
        pad_sentences[l] = pad_sentence([b[0][l] for b in batch], max_len, 0 if l != "input_ids" else pad_token_id)
    for l in batch[0][1]:
        pad_labels[l] = pad_sentence([b[1][l] for b in batch], max_len, -100)
    return pad_sentences, pad_labels

class KanbunDataset(Dataset):
    def __init__(self, path:str, sentence:str, labels:List[str], tokenizer:BertTokenizerFast, label_map:Dict[str, Dict[str, int]], device):
        super().__init__()
        self.sentence = sentence
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        with open(path, "r", encoding = "utf-8") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx][self.sentence]
        labels = {}
        for l in self.labels:
            if l in ("okurigana", "particle", "position"):
                labels[l] = torch.tensor([-100] + [self.label_map[l][c[l]] for c in self.data[idx]["characters"]] + [-100], dtype = torch.long).to(self.device)
            else:
                labels[l] = torch.tensor([-100] + [self.label_map[l][i] for i in self.data[idx][l]] + [-100], dtype = torch.long).to(self.device)

        sentence_tokenized = self.tokenizer(sentence, return_tensors = "pt")
        for i in sentence_tokenized:
            sentence_tokenized[i] = sentence_tokenized[i].squeeze(0).to(self.device)
        return sentence_tokenized, labels

def map_prediction(sentence:str, prediction:List[int], label_map:Dict[str, Dict[int, str]], label_type:str):
    return [label_map[label_type][p] for p in prediction[1:len(sentence) + 1]]

def decode_prediction(path:str, prediction:List[List[List[int]]], label_map:Dict[str, Dict[int, str]]):
    mapped_sentence = []

    with open(path, "r", encoding = "utf-8") as file:
        data = json.load(file)

    for i in range(len(data)):
        mapped_sentence.append([map_prediction(data[i]["traditional_chinese"], prediction[0][i], label_map, "okurigana"),
                                map_prediction(data[i]["traditional_chinese"], prediction[1][i], label_map, "particle"),
                                map_prediction(data[i]["traditional_chinese"], prediction[2][i], label_map, "position"),])

    return mapped_sentence

def character_mark(path:str, prediction:List[List[List[str]]]):
    characters = []
    
    with open(path, "r", encoding = "utf-8") as file:
        data = json.load(file)

    for i in range(len(data)):
        characters.append({"characters":[{"character":data[i]["japanese"][c] ,"okurigana":prediction[i][0][c], "particle":prediction[i][1][c], "position":prediction[i][2][c]} for c in range(len(data[i]["japanese"]))]})

    return characters