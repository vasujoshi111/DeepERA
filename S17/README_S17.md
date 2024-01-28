# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP, Transformers, Pytorch Lightning

## S17
**Objective: Train encoder only(BERT), decoder only(GPT) and ViT model.**

BERT (Bidirectional Encoder Representations from Transformers): BERT is a transformer-based machine learning model developed by Google. It is designed to understand the context of words in a sentence by training on a large corpus of text in an unsupervised manner. BERT can be fine-tuned for various natural language processing (NLP) tasks, such as text classification, question answering, and named entity recognition.

Please refer this notebook: [Bert Model training](./bert_encoder_only.ipynb)

GPT (Generative Pre-trained Transformer): GPT is another transformer-based model developed by OpenAI. Unlike BERT, which is bidirectional, GPT is autoregressive, meaning it generates text one token at a time from left to right. GPT is trained on a large corpus of text and can be fine-tuned for various tasks, including text generation, language translation, and summarization.

Please refer this notebook: [GPT Model training](./gpt_encoder_only.ipynb)

ViT (Vision Transformer): ViT is a transformer-based model developed primarily for computer vision tasks. Unlike traditional convolutional neural networks (CNNs), which operate on fixed-size grids of image patches, ViT treats images as sequences of tokens and processes them using transformer blocks. ViT has shown promising results on image classification tasks and can be applied to other vision tasks like object detection and segmentation.

Please refer this notebook: [ViT Model training](./vit_model.ipynb)

Please go through the code and comments to get deeper intuitions.