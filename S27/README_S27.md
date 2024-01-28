# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP, Transformers, Pytorch Lightning

## S21
**Objective: Training Microsoft-Phi2 using QLora(Quantization and Low-Rank Adapters) for QA.**

QLoRA paper was introduced by Rim Dettmers on 23rd May 2023, where they introduced the concept of 4-bit quantization "with" LoRA. 

1. We convert all weights of a LLM to a 4-bit format and fine-tune the 4-bit LLM. A 4-bit integer can have a maximum of 16 values, and a 32-bit nearly 1076! We'll see later how we map these.

2. We use our PEFT library (for LoRA) and use our adapter concept. (Remember we only train our adapter here). Adapters would be in 32-bit (or others)

3.We dequantize the model for inferencing and backpropagate. For updating our Adapter, we use proper 32-bit

HuugingFace App: https://huggingface.co/spaces/Vasudevakrishna/Qlora_Phi2_Model

Please go through the code and comments to get deeper intuitions.