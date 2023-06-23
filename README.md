# Surveying Large Language Models

Code to reproduce the experiments of the paper [Questioning the Survey Responses of Large Language Models](https://arxiv.org/abs/2306.07951).

We survey language models with the American Community Survey
([data dictionary](forms/acs2019_data_dict.txt)). The experiment results can be downloaded from 
[here](https://keeper.mpdl.mpg.de/d/b8090e1c552d45cebb68/). All figures and tables in the paper can be reproduced using the 
Jupyter notebook [figures.ipynb](figures.ipynb).

### Running the experiments

The relevant files to reproduce the experiments are:

 * [script_fill_individual.py](experiments/script_fill_individual.py): obtain language models' responses to individual survey
questions (Section 3, Section 4, Appendix B, Appendix D, Appendix F), for language models from HugginFace's model hub.
 * [script_fill_openai.py](experiments/script_fill_openai.py): obtain GPT-3's responses to individual survey questions (Section 
3, Section 4, Appendix B, Appendix D).
 * [script_fill_sequential.py](experiments/script_fill_sequential.py): sample language models' responses to entire survey 
questionnaires, where questions are presented sequentially while keeping previous answers in-context (Section 5).
 * [signal_test.py](experiments/signal_test.py): perform the signal test on the model generated data (Section 5, Appendix E).
 * [script_randomization_tests.py](experiments/script_randomization_tests.py): additional answer choice
randomization experiments (Appendix C).
