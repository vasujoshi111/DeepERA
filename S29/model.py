import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from transformers import CLIPVisionModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from tqdm import tqdm
import os, peft


class CustomClipPhi2(nn.Module):
    def __init__(self,tokenizer, phi2_model_name, clip_model_name, clip_embed=768, phi_embed=2560):
        super().__init__()

        self.tokenizer = tokenizer
        # These two models are not finetuned
        # pretrained Microsoft phi2 model
        self.phi2_model = AutoModelForCausalLM.from_pretrained(phi2_model_name,torch_dtype=torch.float32, trust_remote_code=True)
        # pretrained OpenAI clip model
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)

        self.EOS_TOKEN_ID    = self.tokenizer.eos_token_id # 50256
        self.IMAGE_TOKEN_ID  = 23903 # token for Comments
        self.clip_embed      = clip_embed
        self.phi_embed       = phi_embed        

        # projection layers
        # Trainable projection layer
        self.projection_layer = torch.nn.Linear(clip_embed, phi_embed)

        # Freeze Weights
        for models in [self.phi2_model, self.clip_model]:
            for param in models.parameters():
                param.requires_grad_(False)

        # load checkpoint weights
        if os.path.exists('./ckpts/model_phase1.pth'):
            self.projection_layer.load_state_dict(torch.load('./ckpts/model_phase1.pth', map_location='cpu'))
            print("Loaded checkpoint weights for projection layer")
        else:
            print("No checkpoint weights for projection layer")
            print("Initializing projection layer with random weights")
            self.projection_layer.weight.data.normal_(mean=0.0, std=0.02)
            self.projection_layer.bias.data.zero_()


    def generate(self, images, tokenizer, config):
        clip_outputs = self.clip_model(**images)
        # remove cls token
        images = clip_outputs.last_hidden_state[:, 1:, :]
        image_embeddings = self.projection_layer(images).to(torch.float16)

        batch_size = images.size()[0]
        predicted_caption = torch.full((batch_size, config.get("max_tokens")), self.EOS_TOKEN_ID, dtype=torch.long, device=config.get('device'))
        img_token_tensor = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1)
        img_token_embeds = self.phi2_model.model.embed_tokens(img_token_tensor.to(image_embeddings.device))
        combined_embeds  = torch.cat([image_embeddings, img_token_embeds], dim=1)

        for pos in range(config.get("max_tokens") - 1):
            model_output_logits = self.phi2_model.forward(inputs_embeds = combined_embeds)['logits']
            predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1)
            predicted_word_token = torch.argmax(predicted_word_token_logits, dim = -1)
            predicted_caption[:, pos] = predicted_word_token.view(1,-1).to('cpu')
            next_token_embeds = self.phi2_model.model.embed_tokens(predicted_word_token)
            combined_embeds   = torch.cat([combined_embeds, next_token_embeds], dim=1)
        return predicted_caption


    def forward(self, images, target_captions):

        batch_size    = target_captions.size()[0]
        target_length = target_captions.size()[1]
        print("---", target_length)

        # clip model output for image
        clip_outputs = self.clip_model(**images) # See this for loading https://huggingface.co/openai/clip-vit-base-patch36
        images = clip_outputs.last_hidden_state[:, 1:, :] # remove CLS token

        # projection layer
        image_embeddings = self.projection_layer(images).to(torch.float16)

        # add comment token from phi2
        img_token_tensor = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1)
        img_token_embeds = self.phi2_model.model.embed_tokens(img_token_tensor.to(image_embeddings.device))
        combined_embeds  = torch.cat([image_embeddings, img_token_embeds], dim=1) # 4,49,2560
        del clip_outputs
        del image_embeddings

        # for loss
        loss = 0
        for pos in range(target_length - 1):
           
            model_output_logits = self.phi2_model.forward(inputs_embeds = combined_embeds)['logits']
            predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1)
            pos_loss = cross_entropy(predicted_word_token_logits.view(-1,predicted_word_token_logits.size(-1)), target_captions[:, pos].contiguous().view(-1), ignore_index=self.EOS_TOKEN_ID,label_smoothing=0.1)
            loss += pos_loss

            predicted_word_token = torch.argmax(predicted_word_token_logits, dim=-1)
            next_token_embeds = self.phi2_model.model.embed_tokens(predicted_word_token) 
            combined_embeds   = torch.cat([combined_embeds, next_token_embeds], dim=1)
        loss = loss / target_length

        # Delete variables to free up memory
        del combined_embeds
        del model_output_logits
        torch.cuda.empty_cache()

        return loss
        

def show_results_for_samples_phase1(model, val_dataloader, tokenizer, config, num_samples = 2):
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            for images, target_captions in val_dataloader:
                images = {'pixel_values': images.to(config.get('device'))}
                target_captions = target_captions.to(config.get('device'))
                target_captions_decoded = tokenizer.batch_decode(target_captions, ignore_index = tokenizer.eos_token_id)
                predicted_captions = model.generate(images,  tokenizer, config)
                predicted_captions_decoded = tokenizer.batch_decode(predicted_captions,ignore_index = tokenizer.eos_token_id)

                for idx, pc in enumerate(predicted_captions_decoded):
                    print(f"{idx} - Target captions: {target_captions_decoded[idx]} \n {'---------------------'*10} \n Predicted_captions:{pc} ")
                break


def validate_model_phase1(model, val_dataloader, tokenizer, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        try:
            for images, target_captions in tqdm(val_dataloader):
                images = {'pixel_values': images.to(config.get('device'))}
                target_captions = target_captions.to(config.get('device'))
                loss = model(images, target_captions)
                total_loss+=loss.item()
            print(f"Validation Loss: {total_loss/len(val_dataloader)}")
        except Exception as e:
            pass
    model.train()

    
def train_model_phase1(model, train_loader, val_dataloader, optimizer, tokenizer, config):
    model.train()

    pbar = tqdm(train_loader)
    for epoch in range(1, config.get("epochs")):
        print(f"Epoch: {epoch}")
        torch.cuda.empty_cache()
        step = 1
        try:
            for idx, (images, target_captions) in enumerate(pbar):
                try:
                    if target_captions.shape[1] >= config.get("max_tokens"):
                        # print(f"Skipping batch {idx} due to long caption")
                        continue 
        
                    images = {'pixel_values': images.to(config.get('device'))}
                    target_captions = target_captions.to(config.get('device'))
        
                    optimizer.zero_grad()
                    loss = model(images, target_captions)
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f"Epoch: {epoch}: Training Loss = {loss.item()}")
                    torch.cuda.empty_cache()
                    step+=1
                    if (step%1000==0):
                        torch.save(model.projection_layer.state_dict(), './ckpts/model_phase1.pth')
                except Exception as e:
                    print(e)
                    continue
                 
            # # save model
            # if ((epoch % 2) == 0):
                # Only save last checkpoint
            validate_model_phase1(model, val_dataloader, tokenizer, config)
            show_results_for_samples_phase1(model, val_dataloader, tokenizer, config)
            torch.save(model.projection_layer.state_dict(), './ckpts/model_phase1.pth')

        except Exception as e:
            print(e)
            continue




######################################## Phase 2 #########################################

class MainQLoraModel(nn.Module):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.clip_model = CLIPVisionModel.from_pretrained(config.get("clip_model_name"))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        phi2_model = AutoModelForCausalLM.from_pretrained(
            config.get("phi2_model_name"),
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        phi2_model.config.use_cache = False

        ## 4 - LORA config

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64

        peft_config = LoraConfig(
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            r = lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "dense",
                "fc1",
                "fc2"
            ]
        )
        self.phi2_model = peft.get_peft_model(phi2_model, peft_config).to(config.get("device"))

        self.EOS_TOKEN_ID    = self.tokenizer.eos_token_id
        self.clip_embed      = config.get("clip_embed")
        self.phi_embed       = config.get("phi_embed")        

        # projection layers
        # Trainable projection layer
        self.projection_layer = torch.nn.Linear(self.clip_embed, self.phi_embed)

        # Freeze Weights
        for models in [self.clip_model]:
            for param in models.parameters():
                param.requires_grad_(False)

        # load checkpoint weights
        if os.path.exists('./ckpts/model_phase2.pth'):
            self.projection_layer.load_state_dict(torch.load('./ckpts/model_phase2.pth', map_location=config.get("device")))
            self.phi2_model.from_pretrained(self.phi2_model,'./ckpts/Qlora_adaptor')
            print("Loaded checkpoint weights for projection layer")
        else:
            # Load weights from phase 1
            self.projection_layer.load_state_dict(torch.load('./ckpts/model_phase1.pth', map_location=config.get("device")))


    def generate(self, tokenizer, config, images = None, ques = None, max_tokens = 100):
        batch_size = 1

        predicted_caption = torch.full((batch_size, max_tokens), self.EOS_TOKEN_ID, dtype=torch.long, device=self.config.get('device'))
        start_iq = self.tokenizer.encode("<iQ>")
        end_iq = self.tokenizer.encode("</iQ>")
        start_iq_embeds = torch.tensor(start_iq).repeat(batch_size, 1)
        end_iq_embeds = torch.tensor(end_iq).repeat(batch_size, 1)
        start_iq_embeds = self.phi2_model.model.model.embed_tokens(start_iq_embeds.to(self.config.get("device")))
        end_iq_embeds = self.phi2_model.model.model.embed_tokens(end_iq_embeds.to(self.config.get("device")))
        questions_embed  = self.phi2_model.model.model.embed_tokens(ques)
        if images is not None:
            clip_outputs = self.clip_model(**images)
            # remove cls token
            images = clip_outputs.last_hidden_state[:, 1:, :]
            image_embeddings = self.projection_layer(images).to(torch.float16)
            combined_embeds  = torch.cat([start_iq_embeds, image_embeddings, questions_embed, end_iq_embeds], dim=1)
        else:
            combined_embeds = torch.cat([start_iq_embeds, questions_embed, end_iq_embeds], dim=1)

        for pos in range(max_tokens - 1):
            model_output_logits = self.phi2_model.forward(inputs_embeds = combined_embeds)['logits']
            predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1)
            predicted_word_token = torch.argmax(predicted_word_token_logits, dim = -1)
            predicted_caption[:, pos] = predicted_word_token.view(1,-1).to('cpu')
            next_token_embeds = self.phi2_model.model.embed_tokens(predicted_word_token)
            combined_embeds   = torch.cat([combined_embeds, next_token_embeds], dim=1)
        return predicted_caption


    def forward(self, images, ques, ans):

        batch_size = ques.size()[0]
        questions  = ques.to(self.config.get("device"))
        answers    = ans.to(self.config.get("device"))
        target_length = ans.size()[1]
        start_iq = self.tokenizer.encode("<iQ>")
        end_iq = self.tokenizer.encode("</iQ>")
        start_iq_embeds = torch.tensor(start_iq).repeat(batch_size, 1)
        end_iq_embeds = torch.tensor(end_iq).repeat(batch_size, 1)
        start_iq_embeds = self.phi2_model.model.model.embed_tokens(start_iq_embeds.to(self.config.get("device")))
        end_iq_embeds = self.phi2_model.model.model.embed_tokens(end_iq_embeds.to(self.config.get("device")))

        questions_embed  = self.phi2_model.model.model.embed_tokens(questions)
        answers_embed    = self.phi2_model.model.model.embed_tokens(answers)

        are_all_zeros = torch.all(images == 0).item()
        if are_all_zeros:
            combined_embeds = torch.cat([start_iq_embeds, questions_embed, end_iq_embeds], dim=1) 
        else:
            images = {'pixel_values': images.to(self.config.get("device"))}
            clip_outputs  = self.clip_model(**images)
            images_embeds = clip_outputs.last_hidden_state[:,1:,:] # remove cls token
            
            # projection
            image_embeds  = self.projection_layer(images_embeds).to(torch.float16)
            combined_embeds = torch.cat([start_iq_embeds, image_embeds, questions_embed, end_iq_embeds], dim=1) 
        
        
        # # for loss
        loss = 0
        for pos in range(target_length - 1):
            model_output_logits = self.phi2_model.forward(inputs_embeds = combined_embeds)['logits']
            predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1)
            pos_loss = cross_entropy(predicted_word_token_logits.view(-1,predicted_word_token_logits.size(-1)), answers[:, pos].contiguous().view(-1), ignore_index=self.EOS_TOKEN_ID,label_smoothing=0.1)
            predicted_word_token = torch.argmax(predicted_word_token_logits, dim=-1)
            next_token_embeds = self.phi2_model.model.model.embed_tokens(predicted_word_token) 
            combined_embeds   = torch.cat([combined_embeds, next_token_embeds], dim=1)
            del model_output_logits
            torch.cuda.empty_cache()
            loss += pos_loss
        loss = loss / target_length

        # Delete variables to free up memory
        del combined_embeds        
        return loss

    # def forward(self, images, ques, ans):

    #     batch_size = ques.size()[0]
    #     questions  = ques.to(self.config.get("device"))
    #     answers    = ans.to(self.config.get("device"))
    #     target_length = ans.size()[1]
    #     start_iq = self.tokenizer.encode("<iQ>")
    #     end_iq = self.tokenizer.encode("</iQ>")
    #     start_iq_embeds = torch.tensor(start_iq).repeat(batch_size, 1)
    #     end_iq_embeds = torch.tensor(end_iq).repeat(batch_size, 1)
    #     start_iq_embeds = self.phi2_model.model.model.embed_tokens(start_iq_embeds.to(self.config.get("device")))
    #     end_iq_embeds = self.phi2_model.model.model.embed_tokens(end_iq_embeds.to(self.config.get("device")))

    #     questions_embed  = self.phi2_model.model.model.embed_tokens(questions)
    #     answers_embed    = self.phi2_model.model.model.embed_tokens(answers)

    #     are_all_zeros = torch.all(images == 0).item()
    #     if are_all_zeros:
    #         combined_embeds = torch.cat([start_iq_embeds, questions_embed, end_iq_embeds, answers_embed], dim=1) 
    #     else:
    #         images = {'pixel_values': images.to(self.config.get("device"))}
    #         clip_outputs  = self.clip_model(**images)
    #         images_embeds = clip_outputs.last_hidden_state[:,1:,:] # remove cls token
            
    #         # projection
    #         image_embeds  = self.projection_layer(images_embeds).to(torch.float16)
    #         combined_embeds = torch.cat([start_iq_embeds, image_embeds, questions_embed, end_iq_embeds, answers_embed], dim=1) 
        
    #     model_output_logits = self.phi2_model.forward(inputs_embeds = combined_embeds)['logits']
    #     # # for loss
    #     loss = 0
    #     for pos in range(target_length - 1):
    #         predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1)
    #         pos_loss = cross_entropy(predicted_word_token_logits.view(-1,predicted_word_token_logits.size(-1)), answers[:, pos].contiguous().view(-1), ignore_index=self.EOS_TOKEN_ID,label_smoothing=0.1)
    #         loss += pos_loss
    #     loss = loss / target_length

    #     # Delete variables to free up memory
    #     del combined_embeds
    #     del model_output_logits
    #     torch.cuda.empty_cache()
    #     return loss

def validate_model_phase2(model, val_dataloader, tokenizer, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # try:
        for images, ques, ans in tqdm(val_dataloader):
            loss = model(images, ques, ans)
            total_loss+=loss.item()
        print(f"Validation Loss: {total_loss/len(val_dataloader)}")
        # except Exception as e:
        #     pass
    model.train()


def train_model_phase2(model, train_loader, val_dataloader, tokenizer, config):
    phi2_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.phi2_model.parameters()), lr=1e-5)
    proj_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.projection_layer.parameters()), lr=1e-5)
    model.phi2_model.train()
    model.projection_layer.train()

    pbar = tqdm(train_loader)
    for epoch in range(1, config.get("epochs")):
        print(f"Epoch: {epoch}")
        torch.cuda.empty_cache()
        step = 1
        try:
            for idx, (images, ques, ans) in enumerate(pbar):
                try:
                    phi2_optim.zero_grad()
                    proj_optim.zero_grad()
                    loss = model(images, ques, ans)
                    loss.backward()
                    phi2_optim.step()
                    proj_optim.step()
                    pbar.set_description(f"Epoch: {epoch}: Training Loss = {loss.item()}")
                    torch.cuda.empty_cache()
                    step+=1
                    if (step%1000==0):
                        torch.save(model.projection_layer.state_dict(), './ckpts/model_phase2.pth')
                        model.phi2_model.save_pretrained('./ckpts/Qlora_adaptor/', save_adapter=True, save_config=True)
                except Exception as e:
                    print("in frp",e)
                    continue
                 
            validate_model_phase2(model, val_dataloader, tokenizer, config)
            torch.save(model.projection_layer.state_dict(), './ckpts/model_phase2.pth')
            model.phi2_model.save_pretrained('./ckpts/Qlora_adaptor/', save_adapter=True, save_config=True)

        except Exception as e:
            print(e)
            continue