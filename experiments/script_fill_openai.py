# Script to fill the ACS form using OpenAI's API

import sys
sys.path.append('.')

import numpy as np

from surveying_llms.fill_openai import fill_naive, fill_adjusted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)  # seed for randomization
    parser.add_argument('--sleep_t', type=float, default=0.01)  # sleep time between queries, comply with API rate limit

    args = parser.parse_args()
    save_name = args.output_dir + model_name

    # Create the folder if it doesn't exist
    last_slash = args.output_name.rfind('/')
    base_folder = args.output_name[:last_slash]
    name = args.output_name[last_slash + 1:]
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    np.random.seed(args.seed)

    from forms.acs2019 import ACS2019
    form = ACS2019()

    fill_naive(form, args.model_name, save_name, sleep_time=args.sleep_t)
    fill_adjusted(form, args.model_name, save_name, sleep_time=args.sleep_t)

    print('Done')

