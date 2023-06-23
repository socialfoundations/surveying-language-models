# A form is a collection of questions (i.e., questionnaire), which are linked together in a directed graph.

import numpy as np
from .questions import MultipleChoiceQ


class Form:
    def __init__(self):
        self.context = ""
        self.questions = {}
        self.first_q = None

    def n_categories_dict(self):
        """ Returns a dictionary of the form (question key, number of answer choices) """
        sorted_q_keys = self.get_sorted_q_keys()
        return {self.questions[qkey].key: len(self.questions[qkey].choices) for qkey in sorted_q_keys}

    # ------------------------------------------------------------------------------------------------
    # Methods to, once the form has been filled, return the answers to each question as a dictionary
    # ------------------------------------------------------------------------------------------------

    def answers(self, allow_nans=False):
        """ Once the form has been filled, return the answers (i.e., codes, order, and char) as a dictionary

        If allow_nans is True, then unanswered questions will be assigned np.nan as an answer
        """
        answers = {}
        question_id = self.first_q
        while question_id != 'end':
            question = self.questions[question_id]
            qkey = question.key
            code, order, char = question.return_answer(not allow_nans)
            answers[qkey] = code
            answers[qkey + '_order'] = order
            answers[qkey + '_char'] = char
            question_id = question.next_question()
        return answers

    # ----------------------------------------------------------------------------------------------------------
    # Functions to randomize, or otherwise alter, the order in which answer choices are presented, or its labels
    # ----------------------------------------------------------------------------------------------------------

    def randomize_order_answers(self):
        """ Randomize the order of answers (e.g., 'Male', 'Female') for each question """
        for question in self.questions.values():
            if isinstance(question, MultipleChoiceQ):
                question.randomize_order_choices()
        return self

    def randomize_order_choices(self):
        """ Randomize the order of choices (e.g., 'A', 'B') for each question """
        for question in self.questions.values():
            if isinstance(question, MultipleChoiceQ):
                question.randomize_order_completions()
        return self

    def set_choice_chars(self, completions):
        for question in self.questions.values():
            if isinstance(question, MultipleChoiceQ):
                question.set_completions(completions)
        return self

    # -------------------------------------------------------
    # Functions to print as a string the form in various ways
    # -------------------------------------------------------

    def print(self):
        """ Return as a string the form as it would be presented to the model """
        text = self.context
        question = self.first_q
        while question != 'end':
            text += self.questions[question].print()
            question = self.questions[question].next_question()
        return text

    def print_data_dict(self):
        """ Return as a string the data dictionary corresponding to the form (i.e. the questions and their answers) """
        text = self.context
        question = self.first_q
        while question != 'end':
            text += self.questions[question].key + '\n'
            text += self.questions[question].print_data_dict() + '\n\n'
            question = self.questions[question].next_question()
        return text

    # -----------------------------------------------------------------
    # Functions to alter the order in which the questions are presented
    # -----------------------------------------------------------------

    def get_sorted_q_keys(self):
        q_keys = list(self.questions.keys())

        # Get the question numbers
        q_ns = {}
        for q_key in q_keys:
            # Question number
            split = q_key.split('_')
            q_ns[q_key] = float(split[1])

            # If there is a letter (e.g., q_14_a), add a small number
            if len(split) > 2:
                q_ns[q_key] += (ord(split[2]) - ord('a') + 1) / 100

        # Sort the keys by the question number
        sorted_keys = sorted(q_keys, key=lambda x: q_ns[x])
        return sorted_keys

    def sort_q_sequential(self):
        """ Questions are sorted such that they follow the order of the survey """
        sorted_keys = self.get_sorted_q_keys()

        # Assign next questions
        for i, q_key in enumerate(sorted_keys[:-1]):
            self.questions[q_key].next_q = sorted_keys[i + 1]
        self.questions[sorted_keys[-1]].next_q = 'end'
        self.first_q = sorted_keys[0]

    def randomize_order_questions(self):
        """ Questions are sorted in a random order """
        q_keys = list(self.questions.keys())
        np.random.shuffle(q_keys)
        for i, q_key in enumerate(q_keys[:-1]):
            self.questions[q_key].next_q = q_keys[i + 1]
        self.questions[q_keys[-1]].next_q = 'end'
        self.first_q = q_keys[0]
        return self

    def move_question_behind(self, q1, q2):
        """ Move the question `q2` behind the question `q1`

        Do q2_prev.next_q <- q2.next_q, q1.next_q <- q2, q2.next_q <- q1_next, such that
            q1_prev, q1, q1_next, ...., q2_prev, q2, q2_next -> q1_prev, q1, q2, q1_next, ...., q2_prev, q2_next
            q2_prev, q2, q2_next, ...., q1_prev, q1, q1_next -> q2_prev, q2_next, ..., q1_prev, q1, q2, q1_next

        Inputs:
        -------
        q1: str, either one of self.questions.keys() or 'start'/'end' for placing `q2` at the start or end of the form
        q2: str, one of self.questions.keys()
        """
        q1_in_questions = q1 in self.questions.keys()
        assert q1 in ['start', 'end'] or q1_in_questions, f"Invalid q1={q1}"
        assert q2 in self.questions.keys(), f"Invalid q2={q2}"
        assert q1 != q2, "The two questions cannot be the same"

        # If q2 should be the first or last question, and it is, then we are done
        if (q1 == 'start' and self.first_q == q2) or (q1 == 'end' and self.questions[q2].next_q == 'end'):
            return self

        # Function to find the question that precedes some target questions `qt`
        def get_qprev(qt):
            for q_key, q in self.questions.items():
                if q.next_q == qt:
                    return q
            raise ValueError(f"The question {qt} is not the next question of any of the form's questions")

        # q2_prev.next_q <- q2.next_q
        q2_next = self.questions[q2].next_q
        if self.first_q == q2:
            self.first_q = q2_next
        else:
            get_qprev(q2).next_q = q2_next

        # q2.next_q <- q1_next
        # q1.next_q <- q2
        if q1 == 'end':  # place `q2` at the very end
            self.questions[q2].next_q = 'end'
            get_qprev('end').next_q = q2
        elif q1 == 'start':  # place `q2` at the start of the form
            self.questions[q2].next_q = self.first_q
            self.first_q = q2
        elif q1_in_questions:  # place `q2` in between some other questions
            self.questions[q2].next_q = self.questions[q1].next_q
            self.questions[q1].next_q = q2

        return self
    
