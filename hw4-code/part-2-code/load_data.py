import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)


    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        if split == 'test':
            data = []
            for nl_query in nl_queries:
                input_text = f"translate English to SQL: {nl_query}"
                encoder_input = tokenizer(
                    input_text,
                    max_length=256,
                    truncation=True,
                    return_tensors='pt'
                )
                
                decoder_initial = tokenizer('<extra_id_0>', return_tensors='pt')
                
                data.append({
                    'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
                    'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
                    'decoder_initial_input_ids': decoder_initial['input_ids'].squeeze(0)
                })
        else:
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
            
            data = []
            for nl_query, sql_query in zip(nl_queries, sql_queries):
                input_text = f"translate English to SQL: {nl_query}"
                encoder_input = tokenizer(
                    input_text,
                    max_length=256,
                    truncation=True,
                    return_tensors='pt'
                )
                
                decoder_output = tokenizer(
                    sql_query,
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                )
                
                decoder_initial = tokenizer('<extra_id_0>', return_tensors='pt')
                
                data.append({
                    'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
                    'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
                    'decoder_output_ids': decoder_output['input_ids'].squeeze(0),
                    'decoder_initial_input_ids': decoder_initial['input_ids'].squeeze(0)
                })
        
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_output_ids = [item['decoder_output_ids'] for item in batch]
    decoder_initial_input_ids = [item['decoder_initial_input_ids'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    initial_decoder_inputs = pad_sequence(decoder_initial_input_ids, batch_first=True, padding_value=PAD_IDX)
    
    decoder_inputs_list = []
    decoder_targets_list = []
    
    for i in range(len(batch)):
        bos_token = decoder_initial_input_ids[i]
        sql_tokens = decoder_output_ids[i]
        
        decoder_input = torch.cat([bos_token, sql_tokens[:-1]])
        decoder_inputs_list.append(decoder_input)
        decoder_targets_list.append(sql_tokens)
    
    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_initial_input_ids = [item['decoder_initial_input_ids'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    initial_decoder_inputs = pad_sequence(decoder_initial_input_ids, batch_first=True, padding_value=PAD_IDX)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x