import torch
import multiprocessing

def get_config_phase1():
    return {
        "data_dir": "./data",
        "clip_model_name": "openai/clip-vit-base-patch16",
        "phi2_model_name": "microsoft/phi-2",
        "train_batch_size": 2,
        "val_batch_size": 1,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epochs": 2,
        "max_tokens": 20,
        "clip_embed": 768,
        "phi_embed": 2560,
        "num_workers": 4, 
        "ckpts": "./ckpts"
    }

def get_config_phase2():
    return {
        "data_dir": "./data",
        "clip_model_name": "openai/clip-vit-base-patch16",
        "phi2_model_name": "microsoft/phi-2",
        "train_batch_size": 2,
        "val_batch_size": 1,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epochs": 2,
        "max_tokens": 50,
        "clip_embed": 768,
        "phi_embed": 2560,
        "num_workers": 8, 
        "ckpts": "./ckpts",
        "vocab_size": 51200
    }