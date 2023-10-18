# The different types of questions (i.e., MultipleChoice, YesOrNo) which comprise forms.

import numpy as np


class Choice:
    """ Each of the choices in a multiple choice question.

    Inputs
    ------
    text: str, the text of the choice as it appears in the questionnaire
    code: str, the code of the choice as it appears in the PUMS data dictionary
    description: str, bullet-point description of the choice, to be appended to the context
    """
    def __init__(self, text, code=None, description=None):
        self.text = text
        self.code = code
        self.description = description


class Condition:
    """ Condition to evaluate whether a given question should be asked or not.

    Inputs
    ------
    cond: Callable, takes as input the dictionary with questions and returns a boolean, if True, then the question is
            asked, otherwise it is marked as NaN
    prev_qs: Tuple[str], the keys corresponding to the questions used to evaluate the condition
    """
    def __init__(self, cond=None, prev_qs=()):
        self.cond = cond
        self.prev_qs = prev_qs

    def __call__(self, questions):
        if self.cond is None:
            return True

        # If any of the necessary questions have not been answered, then do not answer this question
        for q in self.prev_qs:
            if not questions[q].has_been_answered():
                return False

        return self.cond(questions)

    def __add__(self, other):  # + operator corresponds to AND
        cond = lambda q: self(q) and other(q)  # and operator
        prev_qs = list(set(self.prev_qs + other.get_prev_qs()))
        return Condition(cond, prev_qs)

    def get_prev_qs(self):
        return self.prev_qs


class SingleTokenQ:
    """ Answer admits a single token, e.g., 'A', 'B', etc.

    This means that the question can be evaluated with a single forward pass of the model.

    Inputs
    ------
    text: str, the text of the question
    key: str, when saving the data dictionary, this is the key that will be used to identify the question
    next_q: str, the key of the next question in the questionnaire
    cond: Condition, evaluates whether the question should be asked or not
    description: str, pertaining to the bullet-point description of the question & answer
    bullet_point_is: bool, whether the bullet point should is of the form `q description` 'is' `q answer`
    """
    def __init__(self, text, key, next_q=None, cond=Condition(), description=None, bullet_point_is=True):
        self.text = text
        self.key = key
        self.next_q = next_q
        self.answer_id = None
        self.cond = cond
        self.description = description
        self.bullet_point_is = bullet_point_is

    def next_question(self):
        assert self.next_q is not None, 'No next question specified'
        return self.next_q

    def get_completions(self):
        """ Return the list of str of possible answers, e.g. ['A', 'B', 'C'] """
        raise NotImplementedError

    def get_probs(self, tokenizer, last_word_probs, prefix=' '):
        """ Return the probabilities corresponding to each answer choice given the model's last-token probabilities

        Inputs
        ------
        tokenizer: transformers.PreTrainedTokenizer, tokenizer used to encode the answer choices
        last_word_probs: np.array with shape (vocabulary size,), last-word probabilities output by the language model
        """
        assert len(last_word_probs.shape) == 1
        answers = self.get_completions()
        answer_tokens = [tokenizer.encode(prefix + answer)[-1] for answer in answers]  # -1 since some tokenizers may prepend
        answer_probs = last_word_probs[answer_tokens]
        answer_probs /= answer_probs.sum()
        return np.array(answer_probs).flatten()

    def has_been_answered(self):
        return self.answer_id is not None

    def sample_uniform_answer(self):
        """ Sample an answer choice uniformly at random """
        self.answer_id = np.random.choice(np.arange(len(self.get_completions())))

    def sample_answer(self, answer_probs):
        """ Sample an answer choice from given some probabilities """
        self.answer_id = np.random.choice(np.arange(answer_probs.size), p=answer_probs)

    def get_answer(self, tokenizer, last_word_probs):
        """ Sample an answer from the completion probabilities """
        answer_probs = self.get_probs(tokenizer, last_word_probs)
        self.sample_answer(answer_probs)


class MultipleChoiceQ(SingleTokenQ):
    """ Multiple choice question formulated as follows:

            Question: text
            A. choices[0].text
            B. choices[1].text
            ...
            Answer:

    Inputs
    ------
    text: str, the text of the question
    choices: List[Choice], each of the possible answers
    """
    def __init__(self, text, choices, **kwargs):
        super().__init__(text, **kwargs)
        self.choices = choices

        # Set the codes of the choices, if not done so already (from 1 to len(choices))
        for i, choice in enumerate(self.choices):
            if choice.code is None:
                choice.code = i+1

        # Default completions are A, B, C, ...
        self.completions = [chr(i + 65) for i in range(len(self.choices))]

    def get_completions(self):
        """ e.g., ['A', 'B', 'C'] """
        return self.completions

    def get_choices(self):
        return {choice.code for choice in self.choices}

    def get_choices_permuted(self, perm):
        return [self.choices[i].code for i in perm]

    def get_n_choices(self):
        return len(self.choices)

    def randomize_order_choices(self):
        """ For instance, permute ['Male', 'Female'] to ['Female, 'Male'] """
        np.random.shuffle(self.choices)

    def randomize_order_completions(self):
        """ For instance, permute ['A', 'B', 'C'] to ['B', 'C', 'A'] """
        np.random.shuffle(self.completions)

    def set_completions(self, completions):
        """ Set completions to be different from 'A', 'B', 'C', ... """
        assert len(completions) >= len(self.choices), 'Must have the same number of choices and completions'
        self.completions = completions[:len(self.choices)]

    def get_answer_char(self):
        """ e.g., ' A' """
        assert self.has_been_answered(), 'Question has not been answered'
        return ' ' + self.completions[self.answer_id]

    def get_question(self):
        """ Construct the prompt for the language model """
        text = 'Question: ' + self.text + '\n'
        for i, choice in enumerate(self.choices):
            text += self.completions[i] + '. ' + choice.text + '\n'
        text += 'Answer:'
        return text

    def get_question_interview(self):
        """ Construct the prompt for the language model """
        text = 'Interviewer: ' + self.text + '\n'
        for i, choice in enumerate(self.choices):
            text += self.completions[i] + '. ' + choice.text + '\n'
        text += 'Me:'
        return text

    def get_question_interview_perm(self, perm):
        """ Construct the prompt for the language model """
        text = 'Interviewer: ' + self.text + '\n'
        choices = [self.choices[i] for i in perm]
        for i, choice in enumerate(choices):
            text += self.completions[i] + '. ' + choice.text + '\n'
        text += 'Me:'
        return text

    def get_question_durmus(self, perm):
        """ Construct the prompt for the language model """
        text = 'User: ' + self.text + '\n\nHere are the options:\n'
        choices = [self.choices[i] for i in perm]
        for i, choice in enumerate(choices):
            text += '(' + self.completions[i] + ') ' + choice.text + '\n'
        text += '\nAssistant: If had to select one of the options, my answer would be ('
        return text

    def get_question_permuted(self, perm):
        """ Construct the prompt, but from a set of choices different to those originally passed

        Inputs
        ------
        choices: List[Int], the permutation to be applied to the choices
        """
        text = 'Question: ' + self.text + '\n'
        choices = [self.choices[i] for i in perm]
        for i, choice in enumerate(choices):
            text += self.completions[i] + '. ' + choice.text + '\n'
        text += 'Answer:'
        return text

    def set_answer(self, answer_code):
        """ Manually set the answer to whichever choice corresponds to the code `answer_code`, e.g. 1 """
        assert 0 < answer_code <= len(self.choices)
        for id_choice, choice in enumerate(self.choices):
            if choice.code == answer_code:
                self.answer_id = id_choice
                return id_choice
        raise ValueError(f"No choice found with code {answer_code}")

    def print(self):
        """ Returns as a string the question prompt + the answer choice """
        answer_text = self.get_answer_char() if self.has_been_answered() else ''
        return self.get_question() + answer_text + '\n\n'

    def print_interview(self):
        text = 'Interviewer: ' + self.text + '\n'
        text += 'Me: '
        assert self.has_been_answered(), 'Question has not been answered'
        answer_text = self.choices[self.answer_id].text
        return text + answer_text + '\n'

    def return_answer(self, force_answer=True):
        """ Returns the answer code, answer order, and char with which the choice was labelled , e.g. 1, 0, 'A'.

        if force_answer is True, then an error will be raised if the answer has not been selected yet
        """
        if self.has_been_answered():
            answer_code = self.choices[self.answer_id].code
            answer_order = self.answer_id  # whether the first, second, etc. choice presented was selected
            answer_char = self.completions[self.answer_id]  # the letter assigned to the choice that was presented
            return answer_code, answer_order, answer_char
        else:
            assert not force_answer, 'Answer has not been selected yet' + self.key
            return np.nan, np.nan, np.nan

    def print_bullet_point(self):
        """ Returns the bullet-point descriptor used to incorporate previous answers into the context """
        assert self.description is not None, 'No description provided'
        assert self.answer_id is not None, 'Answer has not been selected yet'
        choice = self.choices[self.answer_id]
        choice_text = choice.text if choice.description is None else choice.description
        if self.description == "":
            is_text = ''
        elif self.bullet_point_is:
            is_text = ' is '
            choice_text = choice_text[0].lower() + choice_text[1:]
        else:
            is_text = ' '
            choice_text = choice_text[0].lower() + choice_text[1:]
        return ' -' + self.description + is_text + choice_text

    def print_data_dict(self):
        """ Returns a string in order to print the data dictionary of a questionnaire """
        text = 'Question: ' + self.text + '\n'
        for i, choice in enumerate(self.choices):
            text += self.completions[i] + '. (code ' + str(choice.code) + ') ' + choice.text + '\n'
        return text


class YesOrNoQ(MultipleChoiceQ):
    """ Multiple-choice question with only two choices: Yes and No. """
    def __init__(self, text, description=None, **kwargs):
        if description is None:
            choices = [Choice('Yes'), Choice('No')]
        else:
            assert len(description) == 2, 'Must provide a description for both choices'
            choices = [Choice('Yes', description=description[0]), Choice('No', description=description[1])]
        super().__init__(text, choices, **kwargs)

    def print_bullet_point(self):
        """ Returns the bullet-point descriptor used to incorporate previous answers into the context """
        assert self.answer_id is not None, 'Answer has not been selected yet'
        choice_des = self.choices[self.answer_id].description
        assert choice_des is not None, 'No description provided for choice'
        return ' -' + choice_des
