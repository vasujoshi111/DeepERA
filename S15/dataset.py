import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class BilingualDataset(Dataset):
    def __init__(self, ds, tockenizer_src, tockenizer_tgt, src_lang, tgt_lang, seq_len=350):
        super().__init__()
        self.ds = ds
        self.tockenizer_src = tockenizer_src
        self.tockenizer_tgt = tockenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tockenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tockenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tockenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        # Trasform the text to tokens
        enc_input_tokens = self.tockenizer_src.encode(src_text).ids
        dec_input_tokens = self.tockenizer_tgt.encode(tgt_text).ids
        # Add the SOS, EOS and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # We will add only <s> and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise Exception("The sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [self.sos_token, 
            torch.tensor(enc_input_tokens, dtype = torch.int64), 
            self.eos_token, 
            torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.int64)],
            dim=0
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [self.sos_token, 
            torch.tensor(dec_input_tokens, dtype = torch.int64), 
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)],
            dim=0
        )

        # Add </s> token
        label = torch.cat(
            [torch.tensor(dec_input_tokens, dtype = torch.int64), 
            self.eos_token, 
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)],
            dim=0
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, seq_len) & (1, 1, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.int64), diagonal=1).type(torch.int)
    return mask==0



class BilingualDataModule(LightningDataModule):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, seq_len, src_lang, tgt_lang, batch_size=32):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Split the dataset into train and val sets
        total_len = len(self.ds)
        val_len = int(0.1 * total_len)  # Adjust the validation split as needed
        train_len = total_len - val_len
        self.train_dataset, self.val_dataset = random_split(self.ds, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(
            BilingualDataset(self.train_dataset, self.tokenizer_src, self.tokenizer_tgt, self.src_lang, self.tgt_lang, self.seq_len),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            BilingualDataset(self.val_dataset, self.tokenizer_src, self.tokenizer_tgt, self.src_lang, self.tgt_lang, self.seq_len),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        encoder_inputs = [item["encoder_input"] for item in batch]
        decoder_inputs = [item["decoder_input"] for item in batch]
        encoder_masks = [item["encoder_mask"] for item in batch]
        decoder_masks = [item["decoder_mask"] for item in batch]
        labels = [item["label"] for item in batch]

        return {
            "encoder_input": pad_sequence(encoder_inputs, batch_first=True),
            "decoder_input": pad_sequence(decoder_inputs, batch_first=True),
            "encoder_mask": pad_sequence(encoder_masks, batch_first=True),
            "decoder_mask": pad_sequence(decoder_masks, batch_first=True),
            "label": pad_sequence(labels, batch_first=True),
            "src_text": tuple([item["src_text"] for item in batch]),
            "tgt_text": tuple([item["tgt_text"] for item in batch])
        }