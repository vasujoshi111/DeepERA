from model import build_transfomer
from dataset import BilingualDataset, casual_mask
from config import get_config, get_weights_file_path
import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import warnings
from tqdm import tqdm
import os
from pathlib import Path
# Huggingface datasets and tockenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint

def greedy_decode(model, source, source_mask, tockenizer_src, tockenizer_tgt, max_len, device):
    sos_idx = tockenizer_tgt.token_to_id("[SOS]")
    eos_idx = tockenizer_tgt.token_to_id("[EOS]")
    # Precompute the encoder output and reuse it every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.shape[1] >= max_len:
            break
        # build mask for target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # compute the decoder output
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # Get next token
        prob = model.project(decoder_output[:, -1])

        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def get_all_sentence(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # Build the tockenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset("opus_books", config["lang_src"]+"-"+config["lang_tgt"], split="train")

    # Build the tockenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])

    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% of the data for training and 10% for validation 
    # train_ds_size = int(len(ds_raw) * 0.9)
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])
    max_eng_length = 150
    filtered_eng_sentences = [sent for sent in ds_raw if len(tokenizer_src.encode(sent["translation"][config["lang_src"]]).ids) <= max_eng_length]

    # Tokenize and filter French sentences
    filtered_fr_sentences = [sent for sent in filtered_eng_sentences if len(tokenizer_tgt.encode(sent["translation"][config["lang_tgt"]]).ids) <= len(tokenizer_src.encode(sent["translation"][config["lang_src"]]).ids) + 10]

    # Keep 90% of the data for training and 10% for validation
    train_ds_size = int(len(filtered_fr_sentences) * 0.9)
    val_ds_size = len(filtered_fr_sentences) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(filtered_fr_sentences, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, seq_len = config["seq_len"], src_lang = config["lang_src"],  tgt_lang = config["lang_tgt"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, seq_len = config["seq_len"], src_lang = config["lang_src"], tgt_lang = ["lang_tgt"])

    # Find the maximum length of the sentences in the source and target sentences
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids 
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of the source sentence: {max_len_src}")
    print(f"Max length of the target sentence: {max_len_tgt}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=val_ds.collate_fn)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transfomer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    return model


def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Make sure the weights folder exists
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Get the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # To use torch.cuda.amp
    scaler = GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-09)
    lr_scheduler = OneCycleLR(optimizer, max_lr=config["lr"], total_steps=config["num_epochs"] * len(train_loader), pct_start=5 / config["num_epochs"] ,anneal_strategy="linear", div_factor=100.0, final_div_factor=100.0, three_phase=True)

    #If the user specified a model to preload, load it
    initail_epoch = 0
    global_step = 0
    if config["preload"]:
        model_file_name = get_weights_file_path(config, config["preload"])
        print(f"PreLoading weights from {model_file_name}")
        state = torch.load(model_file_name)
        model.load_state_dict(state["model_state_dict"])
        initail_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        print(f"PreLoaded weights from {model_file_name}")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)

    for epoch in range(initail_epoch, config["num_epochs"]):
        # try:
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader, desc = f"Processing epoch {epoch}", position=0, leave=True)
        for batch in tqdm(batch_iterator):
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)
            # Run te tenosrs through encoder, decode and projection layer

            with autocast():
                encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
                
                decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)    # (batch_size, seq_len, d_model)
                proj_output = model.project(decoder_output)   # (batch_size, seq_len, vocab_size)
                # Compare the output with label
                label = batch["label"].to(device) # (batch_size, seq_len)
                # Compute the loss using simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # Log the loss
            writer.add_scalar("Training Loss: ", loss.item(), global_step)
            writer.flush()
            # Backpropagate the loss
            scaler.scale(loss).backward()
            # scaler.step(optimizer)
            optimizer.step()
            lr_scheduler.step()
            
            # Clear the gradients
            optimizer.zero_grad(set_to_none=True)
            # Increment the global step
            global_step += 1
        # except Exception as e:
        #     continue
        # Run the validation at the end of each epoch
        run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, print_msg, global_step, num_examples=2)

        model_file_name = get_weights_file_path(config, f"{epoch:2d}")

        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            }, model_file_name)

        print(f"Saved weights to {model_file_name}")
    
    writer.close()



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    try:
        #Get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, set it to 80
        console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)
            # Check that batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print_msg(f"-" * console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Target: {target_text}")
            print_msg(f"Predicted: {model_out_text}")
            if count >= num_examples:
                break
    # Log the validation results
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("Validation CER: ", cer, global_step)
        writer.flush()
        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("Validation WER: ", wer, global_step)
        writer.flush()
        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("Validation BLEU: ", bleu, global_step)
        writer.flush()


def print_msg(msg):
    print(msg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)