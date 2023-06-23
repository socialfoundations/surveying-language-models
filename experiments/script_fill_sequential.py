# Script for sampling entire questionnaires sequentially (Section 5)

import sys
sys.path.append('.')

import numpy as np
import torch

from surveying_llms.fill import BatchForms
from surveying_llms.utils import load_tokenizer_model, move_tmp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--move_tmp', action='store_true')  # faster loads in our internal cluster
    parser.add_argument('--bp', action='store_true')  # bullet point prompting approach (otherwise q&a)

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

    from forms.acs2019 import ACS2019

    randomize_answers = lambda form: form.randomize_order_answers()
    batch_forms = BatchForms(ACS2019, args.n_samples, model_batch_size=args.batch_size,
                             model_context_size=context_window, apply_f=randomize_answers)

    batch_forms.fill(tokenizer, model, ask_sequentially=True, bullet_point=args.bp)
    batch_forms.save_answers(args.output_name + '_s' + str(args.seed) + '.csv', allow_nans=True)

    print('Done')