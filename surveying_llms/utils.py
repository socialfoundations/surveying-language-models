# Util functions to load the language models and tokenizers

import os
import shutil
import torch
import transformers


def add_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)

        input_embeddings[-1:] = input_embeddings_avg
        output_embeddings[-1:] = output_embeddings_avg


def load_tokenizer_model(model_name):
    if 'llama' in model_name or 'alpaca' in model_name or 'vicuna' in model_name or 'koala' in model_name:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name,
                                                               cache_dir="/tmp")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                               cache_dir="/tmp")

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                              torch_dtype=torch.float16,
                                                              cache_dir="/tmp",
                                                              trust_remote_code=True,
                                                              device_map='auto')

    # model.cuda()
    add_pad_token(tokenizer, model)

    return tokenizer, model


def move_tmp(model_dir):
    """ Move the model files to /tmp for faster loads in our internal cluster """
    if os.path.isdir(model_dir):
        tmp_dir = '/tmp/' + model_dir.split('/')[-1]
        if not os.path.isdir(tmp_dir):
            print('Moving model to /tmp/, specifically ', tmp_dir)
            shutil.copytree(model_dir, tmp_dir)
        return tmp_dir
    return model_dir