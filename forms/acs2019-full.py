# Questionnaire based on the 2019 American Community Survey
# https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2019/quest19.pdf

import sys
sys.path.append('.')

from surveying_llms.form import Form
from surveying_llms.questions import Choice, Condition, MultipleChoiceQ, YesOrNoQ


class ACS2019(Form):
    def __init__(self, context=""):
        self.context = context
        self.appendix = ""

        q_3 = MultipleChoiceQ(
            "What is this person's sex?",
            [Choice("Male", code=1),
             Choice("Female", code=2)],
            key='SEX',
            description='Sex',
        )

        is_female = Condition(lambda qs: qs['q_3'].return_answer()[0] == 2, ['q_3'])

        q_4 = MultipleChoiceQ(
            "What is this person's age?",
            [Choice("Under 5 years"),
             Choice("5 to 15 years"),
             Choice("16 to 30 years"),
             Choice("31 to 40 years"),
             Choice("41 to 50 years"),
             Choice("51 to 64 years"),
             Choice("65 years and over")],
            key='AGER',
            description='Age',
        )

        above_5yo = Condition(lambda qs: qs['q_4'].return_answer()[0] > 1, ['q_4'])
        above_15yo = Condition(lambda qs: qs['q_4'].return_answer()[0] > 2, ['q_4'])
        above_30yo = Condition(lambda qs: qs['q_4'].return_answer()[0] > 3, ['q_4'])
        below_51yo = Condition(lambda qs: qs['q_4'].return_answer()[0] < 6, ['q_4'])

        q_5 = YesOrNoQ(
            "Is this person of Hispanic, Latino, or Spanish origin?",
            key='HISPR',
            description=['Hispanic, Latino, or Spanish origin',
                         'Not Hispanic, Latino, or Spanish origin'],
        )

        q_6 = MultipleChoiceQ(
            "What is this person's race?",
            [Choice('White alone'),
             Choice("Black or African American alone"),
             Choice("American Indian or Alaska Native alone"),
             Choice("Asian alone"),
             Choice("Some other race alone"),
             Choice("Two or more races")],
            key='RAC1PR',
            description="Race",
        )

        q_7 = MultipleChoiceQ(
            "Where was this person born?",
            [Choice("In the United States"),
             Choice("Outside of the United States")],
            key='NATIVITY',
            description="Born",
            bullet_point_is=False,
        )

        q_8 = MultipleChoiceQ(
            "Is this person a citizen of the United States?",
            [Choice("Yes, born in the United States",
                    description="United States citizen, born in the United States"),
             Choice("Yes, born in Puerto Rico, Guam, the U.S. Virgin Islands, or Northern Marianas",
                    description="United States citizen, born in Puerto Rico, Guam, the U.S. Virgin Islands, or "
                                "Northern Marianas"),
             Choice("Yes, born abroad of U.S. citizen parent or parents",
                    description="United States citizen, born abroad of U.S. citizen parent or parents"),
             Choice("Yes, U.S. citizen by naturalization",
                    description="United States citizen by naturalization"),
             Choice("No, not a U.S. citizen",
                    description="Not a United States citizen")],
            key='CIT',
            description="",
            bullet_point_is=False,
        )
    
        q_10 = MultipleChoiceQ(
            "At any time in the last 3 months, has this person attended school or college?",
            [Choice("No, has not attended in the last 3 months",
                    description="Has not attended school or college in the last 3 months"),
             Choice("Yes, public school, public college",
                    description="Attended public school or public college in the last 3 months"),
             Choice("Yes, private school, private college, home school",
                    description="Attended private school, private college, or home school in the last 3 months")],
            key='SCH',
            cond=above_5yo,
            description="",
            bullet_point_is=False,
        )

        # this is new
        q_10_b = MultipleChoiceQ(
            "What grade or level of school is this person attending?",
            [Choice("Nursery school, preschool"),
                Choice("Kindergarten"),
                Choice("Grade 1"),
                Choice("Grade 2"),
                Choice("Grade 3"),
                Choice("Grade 4"),
                Choice("Grade 5"),
                Choice("Grade 6"),
                Choice("Grade 7"),
                Choice("Grade 8"),
                Choice("Grade 9"),
                Choice("Grade 10"),
                Choice("Grade 11"),
                Choice("Grade 12"),
                Choice("College undergraduate years (freshman to senior)"),
                Choice("Graduate or professional school beyond a bachelor's degree")],
            key='SCHG',
        )

        q_11 = MultipleChoiceQ(
            "What is this person's highest grade or level of school completed?",
            [Choice("No schooling completed"),
             Choice("Nursery or preschool through grade 12"),
             Choice("High school graduate"),
             Choice("College or some college"),
             Choice("After bachelor's degree")],
            key='SCHLR',
            cond=above_5yo,
            description="Highest grade or level of school completed",
        )

        q_14_a = YesOrNoQ(
            "Does this person speak a language other than English at home?",
            key='LANX',
            cond=above_5yo,
            description=["Speaks a language other than English at home",
                         "Does not speak a language other than English at home"],
        )

        speaks_other_language = Condition(lambda qs: qs['q_14_a'].return_answer()[0] == 1, ['q_14_a'])

        q_14_c = MultipleChoiceQ(
            "How well does this person speak English?",
            [Choice("Very well"),
             Choice("Well"),
             Choice("Not well"),
             Choice("Not at all")],
            key='ENG',
            cond=speaks_other_language,
            description="Speaks English",
            bullet_point_is=False,
        )

        q_16 = YesOrNoQ(
            "Is this person currently covered by any health insurance or health coverage plan?",
            key='HICOV',
            description=["Covered by some health insurance or health coverage plan",
                         "Not covered by any health insurance or health coverage plan"],
        )

        # this is new
        q_17_a = YesOrNoQ(
            "Is there a premium for this person's health insurance or health coverage plan?",
            key='HIPR',
        )

        # this is new
        q_17_b = YesOrNoQ(
            "Does this person or another family member receive a tax credit or subsidy based on family income " \
                "to help pay the premium of this person's health insurance or health coverage plan?",
            key='HITC',
        )

        q_18_a = YesOrNoQ(
            "Is this person deaf or does he/she have serious difficulty hearing?",
            key='DEAR',
            description=["Deaf or has serious difficulty hearing",
                         "Not deaf"],
        )

        q_18_b = YesOrNoQ(
            "Is this person blind or does he/she have serious difficulty seeing even when wearing glasses?",
            key='DEYE',
            description=["Blind or has serious difficulty seeing even when wearing glasses",
                         "Not blind"],
        )

        # this is new
        q_19_a = YesOrNoQ(
            "Because of a physical, mental, or emotional condition, does this " \
            "person have serious difficulty concentrating, " \
            "remembering, or making decisions?",
            key='DREM',
        )
        
        # this is new
        q_19_b = YesOrNoQ(
            "Because of a physical, mental, or emotional condition, does this person have serious difficulty walking or climbing stairs?",
            key='DPHY',
        )
        
        # this is new
        q_19_c = YesOrNoQ(
            "Because of a physical, mental, or emotional condition, does this person have difficulty dressing or bathing?",
            key='DDRS',
        )
        
        # this is new
        q_20 = YesOrNoQ(
            "Because of a physical, mental, or emotional condition, does this person have difficulty doing errands alone " \
                "such as visiting a doctor's office or shopping?",
            key='DOUT',
        )

        q_21 = MultipleChoiceQ(
            "What is this person's marital status?",
            [Choice("Now married"),
             Choice("Widowed"),
             Choice("Divorced"),
             Choice("Separated"),
             Choice("Never married")],
            key='MAR',
            cond=above_15yo,
            description="",
        )

        # this is new
        q_22_a = YesOrNoQ(
            "In the past 12 months, did this person get married?",
            key='MARHM',
        )

        # this is new
        q_22_b = YesOrNoQ(
            "In the past 12 months, did this person get widowed?",
            key='MARHW',
        )

        # this is new
        q_22_c = YesOrNoQ(
            "In the past 12 months, did this person get divorced?",
            key='MARHD',
        )

        # this is new
        q_23 = MultipleChoiceQ(
            "How many times has this person been married?",
            [Choice("Once"),
             Choice("Two times"),
             Choice("Three or more times")],
            key='MARHT',
        )

        q_25 = YesOrNoQ(
            "In the past 12 months, has this person given birth to any children?",
            key='FER',
            cond=above_15yo+below_51yo+is_female,
            description=["Gave birth in the past 12 months",
                         "Did not give birth in the past 12 months"],
        )

        q_26 = YesOrNoQ(
            "Does this person have any of his/her own grandchildren under the age of 18 living in this house or "
            "apartment?",
            key='GCL',
            cond=above_30yo,
            description=["Some of his/her grandchildren under the age of 18 live in this house or apartment",
                         "None of his/her grandchildren under the age of 18 live in this house or apartment"],
        )
        
        # this is new
        q_26_b = YesOrNoQ(
            "Is this grandparent currently responsible for most of the basic needs of any grandchildren under the age of 18 "
            "who live in their house or apartment?",
            key='GCR',
        )
        
        # this is new
        q_26_c = MultipleChoiceQ(
            "How long has this grandparent been responsible for most of the basic needs of any grandchildren under the age of 18 " \
                "who live in their house or apartment?",
            [Choice("Less than 6 months"),
                Choice("6 to 11 months"),
                Choice("1 or 2 years"),
                Choice("3 or 4 years"),
                Choice("5 or more years")],
            key='GCM',
        )
        

        q_27 = MultipleChoiceQ(
            "Has this person ever served on active duty in the U.S. Armed Forces, Reserves, or National Guard?",
            [Choice("Never served in the military", code=4,
                    description="Never served in the military"),
             Choice("Only on active duty for training in the Reserves or National Guard", code=3,
                    description="On active duty for training in the Reserves or National Guard"),
             Choice("Now on active duty", code=1,
                    description="Now on active duty in the U.S. Armed Forces, Reserves, or National Guard"),
             Choice("On active duty in the past, but not now", code=2,
                    description="On active duty in the past, but not now in the U.S. Armed Forces, Reserves, or "
                                "National Guard")],
            key='MIL',
            cond=above_15yo,
            description="",
        )
        
        # this is new
        q_29_a = YesOrNoQ(
            "Does this person have a VA service-connected disability rating?",
            key='DRATX',
        )

        # this is new        
        q_29_b = MultipleChoiceQ(
            "What is this person's service-connected disability rating?",
            [Choice("0 percent"),
             Choice("10 or 20 percent"),
             Choice("30 or 40 percent"),
             Choice("50 or 60 percent"),
             Choice("70 percent or higher")],
            key='DRAT',
        )
        
        q_30 = YesOrNoQ(
            "Last week, did this person work for pay at a job (or business)?",
            key='WRK',
            cond=above_15yo,
            description=["Worked for pay at a job or business last week",
                         "Did not work for pay at a job or business last week"],
        )

        # this is new
        q_30_b = YesOrNoQ(
            "Last week, did this person do ANY work for pay, even for as little as one hour?",
            key='WRKP',
        )

        q_31 = MultipleChoiceQ(
            "Which of the following best describes this person's current work status?",
            [Choice("Civilian employed, at work"),
             Choice("Civilian employed, with a job but not at work"),
             Choice("Unemployed"),
             Choice("Armed Forces, at work"),
             Choice("Armed Forces, with a job but not at work"),
             Choice("Not in the labor force")],
            key='ESR',
            cond=above_15yo,
            description="",
        )

        is_employed_at_work = Condition(lambda qs: qs['q_31'].return_answer()[0] in [1, 3], ['q_31'])

        q_32 = MultipleChoiceQ(
            "How did this person usually get to work last week?",
            [Choice("Car, truck, or van"),
             Choice("Bus"),
             Choice("Subway or elevated rail"),
             Choice("Long-distance train or commuter rail"),
             Choice("Light rail, streetcar, or trolley"),
             Choice("Ferryboat"),
             Choice("Taxicab"),
             Choice("Motorcycle"),
             Choice("Bicycle"),
             Choice("Walked"),
             Choice("Worked from home"),
             Choice("Other method")],
            key='JWTRNS',
            cond=is_employed_at_work,
            description="Last week usually got to work by",
            bullet_point_is=False,
        )

        # this is new
        q_36_a = YesOrNoQ(
            "Last week, was this person on layoff from a job?",
            key='NWLA',
        )

        # this is new
        q_36_b = YesOrNoQ(
            "Last week, was this person temporarily absent from a job or business?",
            key='NWAB',
        )

        # this is new
        q_36_c = YesOrNoQ(
            "Has this person been informed that he or she will be recalled to work within the next 6 months or been given a date to return to work?",
            key='NWRE',
        )

        # this is new
        q_37 = YesOrNoQ(
            "During the last 4 weeks, has this person been actively looking for work?",
            key='NWLK',
        )
        
        # this is new
        q_38 = MultipleChoiceQ(
            "Last week, could this person have started a job if offered one, or returned to work if recalled?",
            [Choice("Yes, could have gone to work"),
                Choice("No, because of own temporary illness"),
                Choice("No, because of all other reasons (in school, etc.)")],
            key='NWAV',
        )            

        q_39 = MultipleChoiceQ(
            "When did this person last work, even for a few days?",
            [Choice("Within the past 12 months"),
             Choice("1 to 5 years ago"),
             Choice("Over 5 years ago or never worked")],
            key='WKL',
            cond=above_15yo,
            description="Last worked",
            bullet_point_is=False,
        )

        worked_past_12_months = Condition(lambda qs: qs['q_39'].return_answer()[0] == 1, ['q_39'])
        worked_past_5_years = Condition(lambda qs: qs['q_39'].return_answer()[0] in [1, 2], ['q_39'])

        # this is new
        q_40_a = YesOrNoQ(
            "During the past 12 months (52 weeks), did this person work every week? Count paid vacation, paid sick leave, and military service as work.",
            key='WKW',
        )

        q_40 = MultipleChoiceQ(
            "During the past 12 months (52 weeks), how many weeks did this person work? Include paid time off and "
            "include weeks when the person only worked for a few hours.",
            [Choice("1 to 13 weeks"),
             Choice("14 to 26 weeks"),
             Choice("27 to 39 weeks"),
             Choice("40 to 47 weeks"),
             Choice("48 to 52 weeks")],
            key='WKWN',
            cond=worked_past_12_months,
            description="During the past 12 months, worked",
            bullet_point_is=False,
        )

        q_41 = MultipleChoiceQ(
            "During the past 12 months, in the weeks worked, how many hours did this person usually work each week?",
            [Choice("Less than 10 hours per week"),
             Choice("10 to 19 hours per week"),
             Choice("20 to 34 hours per week"),
             Choice("35 to 44 hours per week"),
             Choice("45 to 59 hours per week"),
             Choice("60 or more hours per week")],
            key='WKHPR',
            cond=worked_past_12_months,
            description="During the past 12 months, usually worked",
            bullet_point_is=False,
        )

        q_42 = MultipleChoiceQ(
            "Which one of the following best describes this person's employment last week or the most recent "
            "employment in the past 5 years?",
            [Choice("For-profit company or organization",
                    description="at a for-profit company or organization"),
             Choice("Non-profit organization",
                    description="at a non-profit organization"),
             Choice("Local government",
                    description="at a local government"),
             Choice("State government",
                    description="at a state government"),
             Choice("Active duty U.S. Armed Forces or Commissioned Corps",
                    description="in active duty, U.S. Armed Forces or Commissioned Corps"),
             Choice("Federal government civilian employee",
                    description="as a federal government civilian employee"),
             Choice("Owner of non-incorporated business, professional practice, or farm",
                    description="as an owner of a non-incorporated business, professional practice, or farm"),
             Choice("Owner of incorporated business, professional practice, or farm",
                    description="as an owner of an incorporated business, professional practice, or farm"),
             Choice("Worked without pay in a for-profit family business or farm for 15 hours or more per week",
                    description="as a worker without pay in a for-profit family business or farm for 15 hours or more"
                                " per week")],
            key='COWR',
            cond=worked_past_5_years,
            description="Currently or last employed",
            bullet_point_is=False,
        )

        # this is new
        q_42_c = MultipleChoiceQ(
            "Which one of the following best describes the industry in which this person was employed this week or the most recent employment in the past 5 years?",
            [Choice("Manufacturing"),
             Choice("Wholesale trade"),
             Choice("Retail trade"),
             Choice("Other (agriculture, construction, service, government, etc.)")],
            key='IND',
        )

        q_44 = MultipleChoiceQ(
            "What was this person's total income during the past 12 months?",
            [Choice('None'),
             Choice('Less than $12,490'),
             Choice("Between $12,490 and $52,000"),
             Choice('Between $52,000 and $120,000'),
             Choice("Above $120,000")],
            key='PINCPR',
            cond=above_15yo,
            description="Total income in the past 12 months",
        )

        # Add all the questions to a dictionary
        self.questions = {}
        variables = list(locals().keys())
        for var_name in variables:
            if var_name.startswith('q_'):
                self.questions[var_name] = locals()[var_name]

        # By default, sort the form in the sequential order of the survey
        self.sort_q_sequential()
