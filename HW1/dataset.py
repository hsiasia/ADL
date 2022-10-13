from random import sample
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        samples.sort(key=lambda sample: len(sample['text'].split()), reverse=True)
        batch = {}
        batch['id'] = [sample['id'] for sample in samples]
        batch['text'] = [sample['text'].split() for sample in samples]
        batch['len'] = torch.tensor([min(len(text), self.max_len) for text in batch['text']])

        batch['text'] = self.vocab.encode_batch(batch['text'], self.max_len)
        batch['text'] = torch.tensor(batch['text'])

        if 'intent' in samples[0].keys():
            batch['intent'] = [self.label2idx(sample['intent']) for sample in samples]
            batch['intent'] = torch.tensor(batch['intent'])
        else:
            batch['intent'] = torch.zeros(len(samples), dtype=torch.long)
        
        # batch['intent'] = torch.tensor(batch['intent'])
        
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        samples.sort(key=lambda sample: len(sample['tokens']), reverse=True)
        batch = {}
        batch['id'] = [sample['id'] for sample in samples]
        batch['tokens'] = [sample['tokens'] for sample in samples]
        batch['len'] = torch.tensor([min(len(token), self.max_len) for token in batch['tokens']])

        batch['tokens'] = self.vocab.encode_batch(batch['tokens'], self.max_len)
        batch['tokens'] = torch.tensor(batch['tokens'])

        if 'tags' in samples[0].keys():
            batch['tags'] = [[self.label2idx(tag) for tag in sample['tags']] for sample in samples]
            batch['tags'] = pad_to_len(batch['tags'], self.max_len, -1)
        else:
            batch['tags'] = [[0] * self.max_len] * len(samples)

        batch['tags'] = torch.tensor(batch['tags'])
        batch['mask'] = batch['tokens'].gt(0).float()

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
