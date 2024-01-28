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
        "num_workers": 32, 
        "ckpts": "./ckpts"
    }

def get_config_phase2():
    return {
        "i150k_json": "./data/llava_instruct_150k.json",
        "QA_datasetName": "OpenAssistant/oasst1",
        "clip_model_name": "openai/clip-vit-base-patch16",
        "phi2_model_name": "microsoft/phi-2",
        "train_batch_size": 1,
        "val_batch_size": 1,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epochs": 2,
        "max_tokens": 20,
        "clip_embed": 768,
        "phi_embed": 2560,
        "num_workers": 1, 
        "ckpts": "./ckpts",
        "vocab_size": 51200
    }