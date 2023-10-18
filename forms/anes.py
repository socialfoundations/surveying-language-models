# Questionnaire based on the 2019 American Community Survey
# https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2019/quest19.pdf

import sys
sys.path.append('.')

from surveying_llms.form import Form
from surveying_llms.questions import Choice, Condition, MultipleChoiceQ, YesOrNoQ


class ANES2016(Form):
    def __init__(self, context=""):
        self.context = context
        self.appendix = ""

        q_1 = MultipleChoiceQ(
            "What is your gender?",
            [Choice("Male"),
             Choice("Female")],
            key='gender',
        )

        q_2 = MultipleChoiceQ(
            "I am going to read you a list of four race categories. What race do you consider yourself to be?",
            [Choice("White"),
             Choice("Black"),
             Choice("Asian"),
             Choice("Hispanic")],
            key='race',
        )

        q_3 = MultipleChoiceQ(
            "What is your age in years?",
            [Choice("Under 5 years"),
             Choice("5 to 15 years"),
             Choice("16 to 30 years"),
             Choice("31 to 40 years"),
             Choice("41 to 50 years"),
             Choice("51 to 64 years"),
             Choice("65 years and over")],
            key='age',
        )

        q_4 = MultipleChoiceQ(
            "What is the highest level of school you have completed, or the highest degree you have received?",
            [Choice("High school"),
             Choice("Some college"),
             Choice("Four-year college degree"),
             Choice("An advanced degree")],
            key='education',
        )

        q_5 = MultipleChoiceQ(
            "Lots of things come up that keep people from attending religious services even if they want to. Thinking"
            " about your life these days, do you ever attend religious services?",
            [Choice("Yes",),
             Choice("No")],
            key='religion',
        )

        q_6 = MultipleChoiceQ(
            "When you see the American flag flying, how does it make you feel?",
            [Choice("Extremely good"),
             Choice("Moderately good"),
             Choice("A little good"),
             Choice("Neither good nor bad"),
             Choice("A little bad"),
             Choice("Moderately bad"),
             Choice("Extremely bad")],
            key='patriotism',
        )

        q_7 = MultipleChoiceQ(
            "Do you ever discuss politics with your family and friends?",
            [Choice("Yes"),
             Choice("No")],
            key='politics_discuss',
        )

        q_8 = MultipleChoiceQ(
            "How interested would you say you are in politics?",
            [Choice("Very interested"),
             Choice("Somewhat interested"),
             Choice("Not very interested"),
             Choice("Not at all interested")],
            key='politics_interest',
        )

        q_9 = MultipleChoiceQ(
            "Did you vote in the 2016 general election?",
            [Choice("Yes"),
             Choice("No")],
            key='voted_2016',
        )

        voted_2016 = Condition(lambda qs: qs['q_9'].return_answer()[0] == 1, ['q_9'])

        q_10 = MultipleChoiceQ(
            "Which presidential candidate did you vote for in the 2016 presidential election?",
            [Choice("Hillary Clinton"),
             Choice("Donald Trump"),
             Choice("Someone else")],
            key='voted_2016_candidate',
            cond=voted_2016,
        )

        q_11 = MultipleChoiceQ(
            "When asked about your political ideology, would you say you are",
            [Choice("Extremely libreral"),
             Choice("Liberal"),
             Choice("Slightly liberal"),
             Choice("Moderate"),
             Choice("Slightly conservative"),
             Choice("Conservative"),
             Choice("Extremely conservative")],
            key='political_ideology',
        )

        q_12 = MultipleChoiceQ(
            "Which would you say best describes your partisan identification. Would you say you are a",
            [Choice("Strong Democrat"),
             Choice("Not very strong Democrat"),
             Choice("Independent, but closer to the Democratic party"),
             Choice("Independent"),
             Choice("Independent, but closer to the Republican party"),
             Choice("Not very strong Republican"),
             Choice("Strong Republican")],
            key='partisan_identification',
        )

        # Add all the questions to a dictionary
        self.questions = {}
        variables = list(locals().keys())
        for var_name in variables:
            if var_name.startswith('q_'):
                self.questions[var_name] = locals()[var_name]

        # By default, sort the form in the sequential order of the survey
        self.sort_q_sequential()
