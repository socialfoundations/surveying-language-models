# Script to evaluate OpinionsQA and GlobalOpinionsQA

import sys
sys.path.append('.')

import numpy as np
import torch

from surveying_llms.fill import fill_naive, fill_adjusted
from surveying_llms.utils import load_tokenizer_model
from forms.opinionsqa import load_opinions_qa, load_global_opinions_qa

if __name__ == "__main__":
    import os
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--use_global', action='store_true')  # for GlobalOpinionsQA, otherwise OpinionsQA
    parser.add_argument('--dset_dir', type=str, default='opinionsqa/')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    context_window = 1024 if 'gpt2' in args.model_dir else 2048

    # If appropriate, move the model to /tmp for faster loads in our internal cluster
    model_dir = move_tmp(args.model_dir) if args.move_tmp else args.model_dir

    # Load the model
    print('Loading model...', model_dir)
    tokenizer, model = load_tokenizer_model(model_dir)

    # Set the random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the folder if it doesn't exist
    last_slash = args.output_name.rfind('/')
    base_folder = args.output_name[:last_slash]
    name = args.output_name[last_slash + 1:]
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    if args.use_global:
        form = load_global_opinions_qa(args.dset_dir)
    else:
        form = load_opinions_qa(args.dset_dir)

    individual_folder = base_folder + '/individual/'
    if not os.path.exists(individual_folder):
        os.makedirs(individual_folder)
    output_name = individual_folder + name

    # Fill the form naively
    print('Filling the form naively...')
    start_t = time.time()
    fill_naive(form, tokenizer, model, args.batch_size, context_window, output_name)
    print('Time taken: ', time.time() - start_t)

    # Fill the form naively
    print('Filling the form with randomized choice ordering...')
    start_t = time.time()
    fill_adjusted(form, tokenizer, model, args.batch_size, context_window, output_name)
    print('Time taken: ', time.time() - start_t)