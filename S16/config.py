from pathlib import Path


def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 30,
        "lr": 10**-3,
        "seq_len": 160,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "Weights",
        "model_base_name": "t_model",
        "preload": False,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_base_name = config["model_base_name"]
    model_file_name = f"{model_base_name}_{epoch}.pt"
    return str(Path(model_folder, model_file_name).resolve().expanduser().absolute()).format(epoch=epoch)