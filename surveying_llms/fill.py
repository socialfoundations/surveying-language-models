# Functions to fill forms: naively, randomizing over answer choice, and sequentially

import itertools
import math
import torch
import pandas as pd
import numpy as np


def query_model_batch(text_inputs, tokenizer, model, context_size):
    """
    Inputs: the inputs to the model as a list of strings
    Returns: model's last token probabilities for each input as a np.array of shape (len(text_inputs), vocab_size)
    """
    # Tokenize
    token_inputs = [tokenizer.encode(text, return_tensors='pt').flatten()[-context_size:] for text in text_inputs]
    id_last_token = [token_input.shape[0] - 1 for token_input in token_inputs]

    # Pad
    tensor_inputs = torch.nn.utils.rnn.pad_sequence(token_inputs,
                                                    batch_first=True,
                                                    padding_value=tokenizer.pad_token_id).cuda()
    attention_mask = tensor_inputs.ne(tokenizer.pad_token_id)

    # Que
    with torch.no_grad():
        logits = model(input_ids=tensor_inputs, attention_mask=attention_mask).logits

    # Probabilities corresponding to the last token after the prompt
    last_token_logits = logits[torch.arange(len(id_last_token)), id_last_token]
    last_token_probs = torch.nn.functional.softmax(last_token_logits, dim=-1).cpu().numpy()
    return last_token_probs


def fill_naive(form, tokenizer, model, batch_size, context_size, save_dir):
    """ Fill the form naively, asking questions individually and presenting answer choices in the order of the ACS """
    context = form.context
    q = form.first_q

    # Ask each question independently
    responses = []
    while q != 'end':
        # Fill batch size
        questions = []
        while len(questions) < batch_size and q != 'end':
            questions.append(form.questions[q])
            q = form.questions[q].next_q

        # Build the text inputs
        text_inputs = [context + question.get_question() for question in questions]

        # Query model
        last_token_probs = query_model_batch(text_inputs, tokenizer, model, context_size)

        # Save the probabilities
        for question, token_probs in zip(questions, last_token_probs):
            # Get the choices and the probabilities of each choice
            choices = question.get_choices()
            choice_probs = question.get_probs(tokenizer, token_probs)

            # Save
            choice_dict = {'var': question.key}
            choice_dict.update({choice: prob for choice, prob in zip(choices, choice_probs)})
            responses.append(choice_dict)

    choice_df = pd.DataFrame(responses)
    choice_df.to_csv(save_dir + '_naive.csv', index=False)


def fill_adjusted(form, tokenizer, model, batch_size, context_size, save_dir, max_perms=5000):
    """ Adjust for randomized choice ordering, questions are asked individually """
    context = form.context

    q = form.first_q
    while q != 'end':
        question = form.questions[q]

        # Get the permutations to be evaluated
        n_choices = question.get_n_choices()
        indices = [i for i in range(n_choices)]
        if math.factorial(n_choices) <= max_perms:  # enumerate all permutations
            permutations = list(itertools.permutations(indices))
        else:  # sample permutations
            permutations = [np.random.permutation(indices) for _ in range(max_perms)]

        choice_codes = [question.get_choices_permuted(perm) for perm in permutations]

        i = 0
        n_permutations = len(permutations)
        choice_probs = np.zeros((n_permutations, n_choices))
        while i < n_permutations:
            n_batch = min(batch_size, n_permutations - i)

            text_inputs = [context + question.get_question_permuted(perm) for perm in permutations[i:i + n_batch]]
            last_token_probs = query_model_batch(text_inputs, tokenizer, model, context_size)

            for j in range(n_batch):
                choice_probs[i + j] = question.get_probs(tokenizer, last_token_probs[j])

            i += n_batch

        choice_df = pd.DataFrame(choice_codes, columns=['c' + str(i) for i in range(n_choices)])
        probs_df = pd.DataFrame(choice_probs, columns=['p' + str(i) for i in range(n_choices)])

        df = pd.concat([choice_df, probs_df], axis=1)
        df.to_csv(save_dir + '_' + question.key + '.csv', index=False)

        q = question.next_question()


def fill_pairwise(form, model, tokenizer, save_name, questions_avoid, context_size, batch_size, max_evals=5000,
                  bullet_point=True):
    """ Pairwise conditionals test, where the answer to the previous question is included in the context """
    questions = list(form.questions.keys())

    # Iterate through each question as a pivot
    for first_q in questions:
        results = []

        # For these, the next question does not make sense in the context of the census
        if first_q in questions_avoid:
            continue

        # First question is the one whose answer is included in the context
        q_first = form.questions[first_q]
        var1 = q_first.key
        n_choices_first = q_first.get_n_choices()

        # Second question is the one being answered
        second_q = q_first.next_question()
        q_second = form.questions[second_q]
        var2 = q_second.key
        n_choices_second = q_second.get_n_choices()

        print(var1, var2)

        # Permutations of the second question's answer choices
        max_perms = max_evals // n_choices_first
        indices = [i for i in range(n_choices_second)]
        if math.factorial(n_choices_second) <= max_perms:  # enumerate all permutations
            permutations = list(itertools.permutations(indices))
        else:  # sample permutations
            permutations = [np.random.permutation(indices) for _ in range(max_perms)]

        # The code corresponding to each choice
        choice_codes = [q_second.get_choices_permuted(perm) for perm in permutations]

        for choice in range(n_choices_first):
            # Set the relevant answer to the first question
            choice_ordering = q_first.set_answer(choice+1)  # +1 because the choices are 1-indexed

            if bullet_point:
                context = "Information about this person:\n" + q_first.print_bullet_point() + "\n\n"
            else:
                context = q_first.print()

            i = 0
            n_permutations = len(permutations)
            choice_probs = np.zeros((n_permutations, n_choices_second))
            while i < n_permutations:
                n_batch = min(batch_size, n_permutations - i)

                text_inputs = [context + q_second.get_question_permuted(perm) for perm in permutations[i:i + n_batch]]
                last_token_probs = query_model_batch(text_inputs, tokenizer, model, context_size)

                for j in range(n_batch):
                    choice_probs[i + j] = q_second.get_probs(tokenizer, last_token_probs[j])

                i += n_batch

            # Register the probabilities
            choice_df = pd.DataFrame(choice_codes, columns=['c' + str(i) for i in range(n_choices_second)])
            probs_df = pd.DataFrame(choice_probs, columns=['p' + str(i) for i in range(n_choices_second)])
            df = pd.concat([choice_df, probs_df], axis=1)
            df['context_code'] = choice
            df['context_ordering'] = choice_ordering
            results.append(df)

        # Save the data
        df = pd.concat(results, axis=0)
        pd.DataFrame(df).to_csv(save_name + '_' + var1 + '-' + var2 + '.csv', index=False)


class BatchForms:
    """ Fill multiple forms (sequentially) by querying the language model in batches.

    Inputs
    ------
    form_class: Form, the form to fill
    n_forms: int, the number of forms to fill
    model_batch_size: int, batch size using when querying the language model
    model_context_size: int, the number of tokens to use as context when querying the language model
    apply_f: Callable, a function to apply to each of the forms after instantiation,
                      e.g., to randomize the order of questions lambda f: f.randomize_order_questions()
    """

    def __init__(self, form_class, n_forms, model_batch_size, model_context_size, apply_f=lambda f: f, **kwargs):
        self.forms = [apply_f(form_class(**kwargs)) for _ in range(n_forms)]
        self.model_batch_size = model_batch_size
        self.model_context_size = model_context_size

    def fill(self, tokenizer, model, ask_sequentially=True, bullet_point=False):
        """  Fill the forms sequentially by querying the language model in batches """
        contexts = [form.context for form in self.forms]
        current_q = [form.first_q for form in self.forms]

        # Indices of the forms that have not been completely filled yet
        not_done = [i for i in range(len(self.forms))]

        ai = 0  # index of the first element in the batch being processed, points to not_done
        while len(not_done) > 0:
            # end of the batch
            ai_max = min(ai + self.model_batch_size, len(not_done))

            text_inputs = []
            for i in not_done[ai:ai_max]:
                context = contexts[i]
                question = self.forms[i].questions[current_q[i]]
                text_inputs.append(context + question.get_question())

            # Query the model and obtain the last token probabilities
            last_token_probs = query_model_batch(text_inputs, tokenizer, model, self.model_context_size)

            # Obtain the answer to each question in the batch
            for pi, qi in enumerate(not_done[ai:ai_max]):
                qs = self.forms[qi].questions  # relevant dictionary of questions
                question = qs[current_q[qi]]  # current question
                question.get_answer(tokenizer, last_token_probs[pi])

                # Update the context adding the answer
                if ask_sequentially:
                    if bullet_point:  # summary context
                        contexts[qi] = "Information about this person:\n" \
                            if contexts[qi] == "" else contexts[qi][:-1]  # delete \n
                        contexts[qi] += question.print_bullet_point() + "\n\n"
                    else:  # q&a context
                        contexts[qi] += question.print()

                # Skip questions until arriving to a question that should be answered
                q = question.next_question()
                if ask_sequentially:  # only check the condition if questions are asked sequentially
                    while (q != 'end') and (not qs[q].cond(qs)):
                        q = qs[q].next_question()
                current_q[qi] = q

            # Remove the forms that have been completely filled
            to_remove = [ai + pi for pi, qi in enumerate(not_done[ai:ai_max]) if current_q[qi] == 'end']
            not_done = [v for i, v in enumerate(not_done) if i not in to_remove]

            # Update the index of the first element in the batch being processed
            ai = (ai_max - len(to_remove))
            if ai >= len(not_done):
                ai = 0

    def get_answers(self, allow_nans=False):
        """ Return the answers to each form as a DataFrame """
        return pd.DataFrame([form.answers(allow_nans=allow_nans) for form in self.forms])

    def save_answers(self, save_path, allow_nans=False):
        """ Save the answers to each form as a .csv file """
        df = self.get_answers(allow_nans=allow_nans)
        df.to_csv(save_path, index=False)
