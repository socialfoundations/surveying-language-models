# Script to naively fill the ACS form and various ablations

import sys
sys.path.append('.')

import numpy as np
import torch

from surveying_llms.fill import fill_naive, fill_adjusted, fill_pairwise
from surveying_llms.utils import load_tokenizer_model, move_tmp


if __name__ == "__main__":
    import os
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--move_tmp', action='store_true')  # faster loads in our internal cluster
    parser.add_argument('--pairwise', action='store_true')  # pairwise conditionals experiment
    parser.add_argument('--ablation', action='store_true')  # 3 additional prompts for the ablation study

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

    from forms.acs2019 import ACS2019
    form = ACS2019()

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

    # Pairwise conditionals
    if args.pairwise:
        # Make new directory
        ablation_folder = base_folder + '/pairwise/'
        if not os.path.exists(ablation_folder):
            os.makedirs(ablation_folder)
        output_name = ablation_folder + name

        questions_avoid = ['q_14_a', 'q_39', 'q_44']  # conditionals are not meaningful for these questions
        fill_pairwise(form, model, tokenizer, output_name, questions_avoid, context_window, args.batch_size)

    # Ablation study
    if args.ablation:
        # Make new directory
        ablation_folder = base_folder + '/ablation/'
        if not os.path.exists(ablation_folder):
            os.makedirs(ablation_folder)
        output_name = ablation_folder + name

        # Different contexts
        contexts = ["Bellow is a question from the American Community Survey.\n\n",
                    "Answer the following question from the American Community Survey.\n\n",
                    "Answer the following question as if you lived at a household in the United States.\n\n"]

        save_names = ['c0', 'c1', 'c2']
        for context, sn in zip(contexts, save_names):
            form = ACS2019(context=context)
            print('Ablation', sn)
            fill_adjusted(form, tokenizer, model, args.batch_size, context_window, output_name + '_' + sn)

        # Form in the second person
        from forms.acs2019_second_person import ACS2019
        form = ACS2019()
        print('Ablation second person')
        fill_adjusted(form, tokenizer, model, args.batch_size, context_window, output_name + '_sp')