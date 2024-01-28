import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


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

        return enc_input_tokens, dec_input_tokens, src_text, tgt_text
        
    
    def collate_fn(self, batch):

        encoder_inputs = []
        decoder_inputs = []
        labels = []
        src_texts = []
        tgt_texts = []
        encoder_masks = []
        decoder_masks = []
        # Get the max length of the batch
        max_len_enc = max([len(x[0]) for x in batch]) + 2 # Add the SOS and EOS token
        max_len_dec = max([len(x[1]) for x in batch]) + 1 # Add the SOS token
        # # Make sure the max length is not bigger than the seq_len
        # # If it is, set it to seq_len
        # # This is done to avoid memory issues
        # # If the max length is bigger than seq_len, the sentence is too long
        # # and we will have to discard it
        # # This is a very simple way to handle this issue
        # max_len_enc = min(max_len_enc, self.seq_len)
        # max_len_dec = min(max_len_dec, self.seq_len)
        
        for i, (enc_input_tokens, dec_input_tokens, src_text, tgt_text) in enumerate(batch):
            # Add the SOS, EOS and padding to each sentence
            enc_num_padding_tokens = max_len_enc - len(enc_input_tokens) - 2
            # We will add only <s> and </s> only on the label
            dec_num_padding_tokens = max_len_dec - len(dec_input_tokens) - 1
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

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            labels.append(label)
            src_texts.append(src_text)
            tgt_texts.append(tgt_text)
            encoder_masks.append((encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int())
            decoder_masks.append((decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)))

        
        return {
            "encoder_input": torch.stack(encoder_inputs, dim = 0), # (seq_len)
            "decoder_input": torch.stack(decoder_inputs, dim = 0), # (seq_len)
            "encoder_mask": torch.stack(encoder_masks, dim = 0), # (1, 1, seq_len)
            "decoder_mask": torch.stack(decoder_masks, dim = 0), # (1, seq_len) & (1, 1, seq_len)
            "label": torch.stack(labels, dim = 0),
            "src_text": src_texts,
            "tgt_text": tgt_texts
        }

def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.int64), diagonal=1).type(torch.int)
    return mask==0

