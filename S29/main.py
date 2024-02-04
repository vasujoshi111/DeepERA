import torch
from dataset import get_data_loaders_phase1, get_data_loaders_phase2
from transformers import AutoTokenizer
from model import CustomClipPhi2, MainQLoraModel, train_model_phase1, train_model_phase2
from configs import get_config_phase1, get_config_phase2

def phase_1():   
    # get config
    config = get_config_phase1() 
    # tokenizer
    tokenizer  = AutoTokenizer.from_pretrained(config.get("phi2_model_name"), trust_remote_code=True)

    # data loaders
    train_dataloader, val_dataloader = get_data_loaders_phase1(config.get("data_dir"), config.get("clip_model_name"), tokenizer, config.get("train_batch_size"), config.get("val_batch_size"), config.get("num_workers"))

    llmModel = CustomClipPhi2(tokenizer, config.get("phi2_model_name"), config.get("clip_model_name"), clip_embed=768, phi_embed=2560).to(config.get("device"))
    print(llmModel)
    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, llmModel.parameters()), lr=1e-3)
    # train model
    train_model_phase1(llmModel, train_dataloader, val_dataloader, optimizer, tokenizer, config)


def phase_2():   
    # get config
    config = get_config_phase2() 
    # tokenizer
    tokenizer  = AutoTokenizer.from_pretrained(config.get("phi2_model_name"), trust_remote_code=True)

    # data loaders
    train_dataloader, val_dataloader = get_data_loaders_phase2(tokenizer, config)

    llmModel = MainQLoraModel(tokenizer, config).to(config.get("device"))
    print(llmModel)
    # train model
    train_model_phase2(llmModel, train_dataloader, val_dataloader, tokenizer, config)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    phase_1()
    phase_2()