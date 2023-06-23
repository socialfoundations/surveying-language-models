# Scripts for various choice randomization tests (Appendix C)

import sys
sys.path.append('.')

import numpy as np
import torch

from surveying_llms.fill import BatchForms
from surveying_llms.utils import load_tokenizer_model, move_tmp


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--move_tmp', action='store_true')  # faster loads in our internal cluster

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

    def run_randomization_test(name, apply_f):
        print('Running test: ', name)
        start_t = time.time()
        batch_forms = BatchForms(ACS2019, args.n_samples, model_batch_size=args.batch_size,
                                 model_context_size=context_window, apply_f=apply_f)
        batch_forms.fill(tokenizer, model, ask_sequentially=False)  # as questions independently
        batch_forms.save_answers(args.output_name + name + '.csv')
        print('Time taken: ', time.time() - start_t)


    abc = [chr(65 + i) for i in range(26)]
    aic = [chr(65 + i) for i in range(26)]
    aic[1] = 'I'
    rsn = ['R', 'S', 'N', 'L', 'O', 'T', 'M', 'P', 'W', 'U', 'Y', 'V']

    rand_answers = lambda form: form.randomize_order_answers()  # (e.g., 'Male', 'Female')
    rand_choices = lambda form: form.randomize_order_choices()  # (e.g., 'A', 'B')
    rand_choic_answ = lambda form: rand_choices(rand_answers(form))  # both

    letters2test = {'abc': abc, 'aic': aic, 'rsn': rsn}
    transforms2test = {'rand_ca': rand_choic_answ}

    for n_l, letters in letters2test.items():
        set_letter = lambda form: form.set_choice_chars(letters)
        for n_t, apply_f in transforms2test.items():
            run_name = '_' + n_l + '_' + n_t
            run_randomization_test(run_name, lambda form: apply_f(set_letter(form)))

    print('Done')