# Surveying Large Language Models

Code to reproduce the experiments of the paper [Questioning the Survey Responses of Large Language Models](https://arxiv.org/abs/2306.07951).

We survey language models with the American Community Survey. The experiment results can be downloaded from 
[here](https://keeper.mpdl.mpg.de/d/b8090e1c552d45cebb68/). 

### Reproducing the figures in the paper

Use the following Jupyter notebooks:

* Main text: [figures.ipynb](figures.ipynb)
* Appendix: [appendix.ipynb](appendix.ipynb)
* Prompt ablations: [prompt-ablations/](prompt-ablations/)
* Survey ablations: [survey-ablations/](survey-ablations/)

### Running the experiments

The relevant files to reproduce the experiments are:

 * [script_fill_individual.py](experiments/script_fill_individual.py): obtain language models' responses to individual survey questions for language models from HugginFace's model hub.
 * [script_fill_openai.py](experiments/script_fill_openai.py): obtain GPT-3's responses to individual survey questions.
 * [script_fill_sequential.py](experiments/script_fill_sequential.py): sample language models' responses to entire survey.
questionnaires, where questions are presented sequentially while keeping previous answers in-context.
 * [discriminator_test.py](experiments/discriminator_test.py): perform the discriminator test on the model generated data.
