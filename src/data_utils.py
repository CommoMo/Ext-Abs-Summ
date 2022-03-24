import json
import os

from torch.utils.data import Dataset, DataLoader
import torch

import logging
logger = logging.getLogger('[System]')


def init_tokenizer(tokenize):
    global tokenizer
    tokenizer = tokenize

def get_dataloader(args, tokenizer, data_type):
    init_tokenizer(tokenizer)
    if data_type == 'train':
        with open(os.path.join(args.data_dir, args.train_file), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        dataset = MultitaskExtAbsSummaryDataset(args, tokenizer, dataset)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
    else:
        with open(os.path.join(args.data_dir, args.valid_file), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        dataset = MultitaskExtAbsSummaryDataset(args, tokenizer, dataset)
        dataloader = DataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=collate_fn)
    
    return dataloader


class MultitaskExtAbsSummaryDataset(Dataset):
    def __init__(self, args, tokenizer, dataset):
        self.args = args
        self.max_length = args.max_seq_length
        self.tokenizer = tokenizer

        self.dataset= dataset
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        ext = data['ext']
        sentences = data['sentences']
        target_text = data['abs']


        # Ext labels
        sentence_ids = [self.tokenizer.tokenize(sent) for sent in sentences]
        ext_ids = [self.tokenizer.tokenize(sent) for sent in ext]
        label_list = []
        for sent_id in sentence_ids:
            if sent_id in ext_ids:
                indices = [1 for _ in range(len(sent_id))]
            else:
                indices = [0 for _ in range(len(sent_id))]
            label_list.append(indices)
        ext_label = sum(label_list, [])
        ext_label = torch.tensor(ext_label[:self.max_length], dtype=torch.float)

        # Abs lables
        abs_label = self.tokenizer.encode(target_text, max_length=self.max_length, truncation=True)
        abs_label.append(self.tokenizer.eos_token_id)

        # decoder_input_ids
        decoder_input_ids = [self.tokenizer.pad_token_id]
        decoder_input_ids += abs_label[:-1]

        # inputs
        content = ' '.join(sentences)
        inputs = self.tokenizer(content, truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        sentence_ids_lengths = [len(tokens) for tokens in sentence_ids]
        if len(ext_label) != sum(sentence_ids_lengths):
            print(content)

        abs_lable = torch.tensor(abs_label)
        decoder_input_ids = torch.tensor(decoder_input_ids)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'decoder_input_ids': decoder_input_ids,
                'ext_label': ext_label,
                'abs_label': abs_lable,
                'content': content,
                'target_text': target_text
                }

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    input_ids = [_['input_ids'] for _ in batch]
    attention_mask = [_['attention_mask'] for _ in batch]
    decoder_input_ids = [_['decoder_input_ids'] for _ in batch]
    ext_labels = [_['ext_label'] for _ in batch]
    abs_labels = [_['abs_label'] for _ in batch]
    content = [_['content'] for _ in batch]
    target_texts = [_['target_text'] for _ in batch]

    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=3)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=3)
    decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=3)
    ext_labels = torch.nn.utils.rnn.pad_sequence(ext_labels, batch_first=True, padding_value=0)
    abs_labels = torch.nn.utils.rnn.pad_sequence(abs_labels, batch_first=True, padding_value=3)

    return input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels, \
        content, target_texts




if __name__ == '__main__':
    from transformers import AutoTokenizer
    import argparse
    from utils import *
    from tqdm import tqdm

    model_path = 'models/kobart'

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='kobart.json')
    parser.add_argument("--output_dir", type=str, default='kobart_ckpt')
    args = parser.parse_args()
    args = init_setting(args)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_loader = get_dataloader(args, tokenizer, 'valid')

    for batch in tqdm(data_loader):
        batch