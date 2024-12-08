hf_models = [
    'gpt2', 'gpt2m', 'gpt2l', 'gpt2xl',
    'gptneo125m', 'gptneo1.3b', 'gptneo2.7b',
    'mpt', 'mpt-chat', 'mpt-instruct',
    'pythia7', 'pythia12', 'dolly12b',
    'llama7b', 'llama13b', 'llama30b', 'llama65b',
    'koala7b', 'koala13b', 'vicuna7b', 'vicuna13b',
    'gptneox', 'gptneoxchat',
    'llama2-7b', 'llama2-13b', 'llama2-70b',
    'llama2-7b-chat', 'llama2-13b-chat', 'llama2-70b-chat',
    'llama3-8b', 'llama3-70b', 'llama3-8b-instruct', 'llama3-70b-instruct',
]

openai_models = [
    'ada', 'babbage', 'curie', 'davinci',
    'text-davinci-001', 'text-davinci-002', 'text-davinci-003',
    'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo-preview'
]

model_names = {
    'gpt2': 'GPT2 110M', 'gpt2m': 'GPT2 355M', 'gpt2l': 'GPT2 774M', 'gpt2xl': 'GPT2 1.5B',
    'gptneo125m': 'GPT NEO 125M', 'gptneo1.3b': 'GPT NEO 1.3B',  'gptneo2.7b': 'GPT NEO 2.7B',
    'mpt': 'MPT 7B', 'mpt-chat': 'MPT Chat 7B', 'mpt-instruct': 'MPT Instruct 7B',
    'pythia7': 'Pythia 7B', 'pythia12': 'Pythia 12B', 'dolly12b': 'Dolly 12B',
    'llama7b': 'LLaMA 7B', 'llama13b': 'LLaMA 13B', 'llama30b': 'LLaMA 30B', 'llama65b': 'LLaMA 65B',
    'llama2-7b': 'Llama 2 7B', 'llama2-13b': 'Llama 2 13B', 'llama2-70b': 'Llama 2 70B',
    'llama2-7b-chat': 'Llama 2 Chat 7B', 'llama2-13b-chat': 'Llama 2 Chat 13B', 'llama2-70b-chat': 'Llama 2 Chat 70B',
    'llama3-8b': 'Llama 3 8B', 'llama3-70b': 'Llama 3 70B',
    'llama3-8b-instruct': 'Llama 3 8B Inst', 'llama3-70b-instruct': 'Llama 3 70B Inst',
    'koala7b': 'Koala 7B', 'koala13b': 'Koala 13B',
    'vicuna7b': 'Vicuna 7B', 'vicuna13b': 'Vicuna 13B',
    'gptneox': 'GPT NeoX 20B', 'gptneoxchat': 'NeoXT Chat 20B',
    'ada': 'GPT3 2.7B', 'babbage': 'GPT3 6.7B', 'curie': 'GPT3 13B', 'davinci': 'GPT3 175B',
    'text-davinci-001': 'text-davinci-001', 'text-davinci-002': 'text-davinci-002', 'text-davinci-003': 'text-davinci-003',
    'gpt-4': 'GPT-4 0613', 'gpt-4-turbo': 'GPT 4 Turbo', 'gpt-3.5-turbo': 'GPT 3.5 Turbo 0613', 'gpt-4-turbo-preview': 'GPT-4 Turbo 0125',
    'uniform': 'Uniform\ndistribution', 'census': 'U.S. census'
}

model_sizes = {  # in billions of parameters
    'gpt2': 0.11, 'gpt2m': 0.355, 'gpt2l': 0.774, 'gpt2xl': 1.5,
    'gptneo125m': 0.125, 'gptneo1.3b': 1.3, 'gptneo2.7b': 2.7,
    'mpt': 6.8, 'mpt-chat': 6.95, 'mpt-instruct': 6.9,
    'pythia7': 6.9, 'pythia12': 12, 'dolly12b': 12,
    'llama7b': 7, 'llama13b': 13, 'llama30b': 30, 'llama65b': 65,
    'llama2-7b': 7.3, 'llama2-13b': 13.3, 'llama2-70b': 70,
    'llama2-7b-chat': 7.4, 'llama2-13b-chat': 13.4, 'llama2-70b-chat': 70.1,
    'llama3-8b': 8, 'llama3-70b': 71,
    'llama3-8b-instruct': 8.1, 'llama3-70b-instruct': 72,
    'koala7b': 7.1, 'koala13b': 13.1, 'vicuna7b': 7.2, 'vicuna13b': 13.2,
    'gptneox': 20, 'gptneoxchat': 20,
    'ada': 2.7, 'babbage': 6.7, 'curie':13, 'davinci': 174.7,
    'text-davinci-001': 174.8, 'text-davinci-002': 174.9, 'text-davinci-003': 175,
    'gpt-4': 178, 'gpt-3.5-turbo': 176, 'gpt-4-turbo-preview': 177,
    'census': 190, 'uniform': 190,  # such that they are always to the right in the plots
}
for key in model_sizes.keys():  # since parameter count is given in billions
    model_sizes[key] *= 1e9

# Instruction-tuned models, with their corresponding base model
instruct = {
    'mpt-chat': 'mpt', 'mpt-instruct': 'mpt',
    'dolly12b': 'pythia12',
    'koala7b': 'llama7b', 'vicuna7b': 'llama7b',
    'koala13b': 'llama13b', 'vicuna13b': 'llama13b',
    'llama2-7b-chat': 'llama2-7b', 'llama2-13b-chat': 'llama2-13b', 'llama2-70b-chat': 'llama2-70b',
    'llama3-8b-instruct': 'llama3-8b', 'llama3-70b-instruct': 'llama3-70b',
    'gptneoxchat': 'gptneox',
    'text-davinci-001': 'davinci', 'text-davinci-002':'davinci', 'text-davinci-003': 'davinci',
    'gpt-4': 'davinci', 'gpt-3.5-turbo': 'davinci',  # not correct
    'gpt-4-turbo-preview': 'davinci'
}