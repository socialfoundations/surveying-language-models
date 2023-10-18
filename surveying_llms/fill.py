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
    print('Saving to...', save_dir + '_naive.csv')
    choice_df.to_csv(save_dir + '_naive.csv', index=False)


def fill_adjusted(form, tokenizer, model, batch_size, context_size, save_dir, max_perms=5000, prompt=None):
    """ Adjust for randomized choice ordering, questions are asked individually """
    context = form.context

    assert (prompt is None) or (prompt in ['interview', 'durmus'])

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

            if prompt == 'interview':
                question_prompt = question.get_question_interview_perm(perm)
            elif prompt == 'durmus':
                question.get_question_durmus(perm)
            else:
                question_prompt = question.get_question_permuted(perm)

            text_inputs = [context + question_prompt for perm in permutations[i:i + n_batch]]
            last_token_probs = query_model_batch(text_inputs, tokenizer, model, context_size)

            for j in range(n_batch):
                choice_probs[i + j] = question.get_probs(tokenizer, last_token_probs[j])

            i += n_batch

        choice_df = pd.DataFrame(choice_codes, columns=['c' + str(i) for i in range(n_choices)])
        probs_df = pd.DataFrame(choice_probs, columns=['p' + str(i) for i in range(n_choices)])

        df = pd.concat([choice_df, probs_df], axis=1)
        df.to_csv(save_dir + '_' + question.key + '.csv', index=False)

        q = question.next_question()

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

    def fill(self, tokenizer, model, ask_sequentially=True, bullet_point=False, interview=False):
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
                question_prompt = question.get_question_interview() if interview else question.get_question()
                text_inputs.append(context + question_prompt)

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
                        new_context = question.print_interview() if interview else question.print()
                        contexts[qi] += new_context

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
