# Load OpinionsQA or GlobalOpinionsQA

import sys
sys.path.append('.')

import os
import ast
import pandas as pd

from surveying_llms.form import Form
from surveying_llms.questions import Choice, MultipleChoiceQ


def load_opinions_qa(dir):
    files = os.listdir(dir)
    form = Form()
    # for each questionnaire
    for file in files:
        if file[0] != '.' and file[-3:] == 'csv':
            print('Loading ' + file)
            df = pd.read_csv(dir+file, delimiter='\t')

            # for each question in the questionnaire
            for i in range(len(df)):
                row = df.iloc[i]
                choices_text = ast.literal_eval(row['options'])
                choices = [Choice(c) for c in choices_text]
                form.questions[row['key']] = MultipleChoiceQ(row['question'], choices, key=row['key'])

    # set the next q parameter
    q_keys = list(form.questions.keys())
    for i in range(len(q_keys)-1):
        form.questions[q_keys[i]].next_q = q_keys[i + 1]

    # set the first q parameter
    form.first_q = q_keys[0]

    # set the last q parameter
    form.questions[q_keys[-1]].next_q = 'end'

    return form


def load_global_opinions_qa(dir):
    files = os.listdir(dir)
    form = Form()
    print(files)
    # for each questionnaire
    for file in files:
        if file[0] != '.' and file[-3:] == 'csv':
            print('Loading ' + file)
            df = pd.read_csv(dir+file)

            # for each question in the questionnaire
            for i in range(len(df)):
                row = df.iloc[i]
                choices_text = ast.literal_eval(row['options'])
                choices = [Choice(c) for c in choices_text]
                form.questions[str(i)] = MultipleChoiceQ(str(row['question']), choices, key=str(i))

    # set the next q parameter
    q_keys = list(form.questions.keys())
    for i in range(len(q_keys)-1):
        form.questions[q_keys[i]].next_q = q_keys[i + 1]

    # set the first q parameter
    form.first_q = q_keys[0]

    # set the last q parameter
    form.questions[q_keys[-1]].next_q = 'end'

    return form