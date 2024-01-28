import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
import pickle
import requests
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np


class ClipDataset(Dataset):
  '''ClipDataset class for loading the CLIP dataset'''
  def __init__(self, coco_data, model_name, tokenizer):

    self.tokenizer  = tokenizer
    self.processor  = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    self.caption_dataset = coco_data
      
  def __len__(self):
    #Return the length of the dataset
    return len(self.caption_dataset)

  def __getitem__(self, idx):
    #Get the image url and caption
    img_url = self.caption_dataset[idx]["image_url"]
    caption = self.caption_dataset[idx]["caption"]

    #Get the image and caption embeddings
    image = Image.open(requests.get(img_url,stream=True).raw)
    width, height = image.size
    new_width  = 224
    new_height = new_width * height // width 
    new_height = 224
    new_width  = new_height * width // height
    image = image.resize((new_width, new_height), Image.LANCZOS)
    image_processed = self.processor(images=image, return_tensors="pt") ['pixel_values']
    image_sqeezed = image_processed.squeeze(0)
    tokenized_caption = self.tokenizer(caption, return_tensors="pt", return_attention_mask=False)
    tokenized_caption_ids = tokenized_caption['input_ids'].squeeze(0)
    return(image_sqeezed , tokenized_caption_ids)
  

def collate_fn_phase1(batch):
    #Unzip the batch
    image_embeddings, captions = zip(*batch)
    #Stack the image embeddings
    image_embeddings_stacked = torch.stack(image_embeddings, dim=0)
    #Pad the captions, padded value is the <eos> token
    captions_padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=50256)
    #Return the stacked image embeddings and padded captions
    return (image_embeddings_stacked, captions_padded)


def get_data_loaders_phase1(data_dir, clip_model_name, tokenizer, train_batch_size, val_batch_size, num_workers):
    # Load the data
    with open(os.path.join(data_dir, 'coco_train.pkl'), 'rb') as fp: 
        train_pkl = pickle.load(fp)
    with open(os.path.join(data_dir, "coco_val.pkl"), "rb") as fp:
        val_pkl = pickle.load(fp)
   # train data loaders
    train_dataloader = DataLoader(ClipDataset(train_pkl, clip_model_name, tokenizer), collate_fn=collate_fn_phase1, batch_size=train_batch_size, num_workers = num_workers, shuffle=True, pin_memory=True)

    # val data loaders
    val_dataloader   = DataLoader(ClipDataset(val_pkl, clip_model_name, tokenizer), collate_fn=collate_fn_phase1, batch_size=val_batch_size, num_workers = num_workers, shuffle=False, pin_memory=True)
    return train_dataloader, val_dataloader

##################################### Phase 2 #########################################

  
class ClipDatasetPhase2(Dataset):
  '''ClipDataset class for loading the CLIP dataset'''
  def __init__(self, data_frame, model_name, tokenizer):

    self.tokenizer  = tokenizer
    self.processor  = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    self.df = data_frame
      
  def __len__(self):
    #Return the length of the dataset
    return len(self.df)

  def __getitem__(self, idx):
    #Get the image url and QAs
    img_url = self.df.ImageUrl[idx[0]]
    que = self.df.Question[idx[0]]
    ans = self.df.Answer[idx[0]]

    print("img_url", img_url)
    print("que", que)
    print("ans", ans)

    #Get the image and caption embeddings
    if img_url is None:
        print("img_url is None")
        image_sqeezed = None
    else:
        image = Image.open(requests.get(img_url,stream=True).raw)
        width, height = image.size
        new_width  = 224
        new_height = new_width * height // width 
        new_height = 224
        new_width  = new_height * width // height
        image = image.resize((new_width, new_height), Image.LANCZOS)
        image_processed = self.processor(images=image, return_tensors="pt") ['pixel_values']
        image_sqeezed = image_processed.squeeze(0)
    que_ids = self.tokenizer(que, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0)
    ans_ids = self.tokenizer(ans, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0)
    return(image_sqeezed , que_ids, ans_ids)
  

def collate_fn_phase2(batch):
    #Unzip the batch
    image_embeddings, ques, ans = zip(*batch)
    #Stack the image embeddings
    if image_embeddings[0] is None:
        image_embeddings_stacked = None
    else:
        image_embeddings_stacked = torch.stack(image_embeddings, dim=0)
    #Pad the QAs, padded value is the <eos> token
    ques_padded = torch.nn.utils.rnn.pad_sequence(ques, batch_first=True, padding_value=50256)
    ans_padded = torch.nn.utils.rnn.pad_sequence(ans, batch_first=True, padding_value=50256)
    #Return the stacked image embeddings and padded QAs
    return (image_embeddings_stacked, ques_padded, ans_padded)


def prep_data(df):
    df_assistant = df[(df.role == "assistant") & (df["rank"] == 0.0)].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["Answer"] = df_assistant["text"].values

    inputs = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)

    df_assistant["Question"] = inputs
    df_assistant["ImageUrl"] = None

    df_assistant = df_assistant[df_assistant.lang == "en"]

    df_assistant = df_assistant[
        ["ImageUrl","Question", "Answer", "message_id"]
    ].rename(columns={"message_id": "Ids"})

    return df_assistant


def get_i150_df(config):
    with open(config.get("i150k_json"), "r") as fp: 
        i150k_json_read = json.load(fp)
    max_tokens = 100
    image_urls = []
    ques_list = []
    ans_list = []
    id_list = []
    for idx, data in enumerate(i150k_json_read):
        image = data['image']
        image_url = 'http://images.cocodataset.org/train2017/' + image
        id_ = data["id"]
        iterator = iter(data['conversations'])
        for i in iterator:
            ques = i
            ans = next(iterator)
            if (len(ques["value"])>100 or len(ans["value"])>max_tokens):
                continue
            if ques["from"] == "human" and ans["from"] == "gpt":
                image_urls.append(image_url)
                ques_list.append(ques["value"].replace("<image>\n","").replace("<image>",""))
                ans_list.append(ans["value"])
                id_list.append(id_)
    df_i150k = pd.DataFrame(list(zip(image_urls, ques_list, ans_list, id_list)),
                  columns =["ImageUrl", "Question", "Answer", "Ids"])
    msk = np.random.rand(len(df_i150k)) < 0.96

    train_df = df_i150k[msk]
    test_df = df_i150k[~msk]
    return train_df, test_df


def get_oas_df(config):
    train_ds, val_ds = load_dataset(config.get("QA_datasetName"), split=["train", "validation"])
    train_df = prep_data(train_ds.to_pandas())
    test_df = prep_data(val_ds.to_pandas())
    return train_df, test_df


def get_data_loaders_phase2(tokenizer, config):

    train_i150k, test_i150k = get_i150_df(config)
    train_oas, test_oas = get_oas_df(config)

    train_df = pd.concat([train_i150k, train_oas]).reset_index(drop=True)
    val_df = pd.concat([test_i150k, test_oas]).reset_index(drop=True)
   # train data loaders
    train_dataloader = DataLoader(ClipDatasetPhase2(train_df, config.get("clip_model_name"), tokenizer), collate_fn=collate_fn_phase2, batch_size=config.get("train_batch_size"), num_workers = config.get("num_workers"), shuffle=True, pin_memory=True)

    # val data loaders
    val_dataloader   = DataLoader(ClipDatasetPhase2(val_df, config.get("clip_model_name"), tokenizer), collate_fn=collate_fn_phase2, batch_size=config.get("val_batch_size"), num_workers = config.get("num_workers"), shuffle=False, pin_memory=True)
    return train_dataloader, val_dataloader