# Script to fill the ACS form using OpenAI's API

import sys
sys.path.append('.')

import numpy as np

from surveying_llms.fill_openai import fill_naive, fill_adjusted, fill_pairwise_conditionals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--naive', action='store_true')  # naive sampling
    parser.add_argument('--adjusted', action='store_true')  # adjusted via randomized choice ordering
    parser.add_argument('--pairwise', action='store_true')  # pairwise test
    parser.add_argument('--seed', type=int, default=0)  # seed for randomization
    parser.add_argument('--sleep_t', type=float, default=0.01)  # sleep time between queries, comply with API rate limit

    args = parser.parse_args()

    model_name = args.model_name
    save_name = args.output_dir + model_name
    seed = args.seed
    sleep_time = args.sleep_t

    np.random.seed(seed)

    from forms.acs2019 import ACS2019
    form = ACS2019()

    if args.naive:
        fill_naive(form, model_name, save_name, sleep_time=sleep_time)

    if args.adjusted:
        fill_adjusted(form, model_name, save_name, sleep_time=sleep_time)

    if args.pairwise:
        questions_avoid = ['q_14_a', 'q_39', 'q_44']
        fill_pairwise_conditionals(form, model_name, save_name, questions_avoid, sleep_time=sleep_time)

    print('Done')

