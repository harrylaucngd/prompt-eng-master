import json
import csv
import random
import os

import _jsonnet

from agents.zero_shot import ZeroShotModel
from agents.expert import ExpertModel
from agents.few_shot import FewShotModel
from agents.zero_shot_cot import ZeroShotCoTModel
from agents.few_shot_cot import FewShotCoTModel
from eval import capability_fn, accuracy_fn


def Model(model_name):
    if model_name in ["GPT-3.5", "GPT-4"]:
        with open('config/api.json', 'r') as api_file:
            all_api = json.load(api_file)
            openai_api = all_api["openai_api"]
        api_key = os.environ.get("OPENAI_API_KEY") or openai_api["api_key"]
        api_base = os.environ.get("OPENAI_API_BASE") or openai_api["api_base"]
        if api_key is None or api_base is None:
            print("The required environment variables OPENAI_API_KEY or OPENAI_API_BASE have not been set. Please set these environment variables first.")
        api = [api_key, api_base]
        with open('config/gpt_configuration.json', 'r') as config_file:
            all_configs = json.load(config_file)
            config = all_configs[model_name]
            return config, api
    elif model_name in ["LLaMA2-7B", "LLaMA2-13B", "LLaMA2-70B"]:   # Set as the same as above. We shall implement them in the future.
        with open('config/llama_configuration.json', 'r') as config_file:
            all_configs = json.load(config_file)
            config = all_configs[model_name]
            return config
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def TestAgent(agent_name):
    if agent_name == 'zero-shot':
        return ZeroShotModel
    elif agent_name == 'expert':
        return ExpertModel
    elif agent_name == 'few-shot':
        return FewShotModel
    elif agent_name == 'zero-shot-CoT':
        return ZeroShotCoTModel
    elif agent_name == 'few-shot-CoT':
        return FewShotCoTModel
    else:
        raise ValueError(f"Unsupported prompt: {agent_name}")


def input_counter(data_name):
    input_dict = {
        "data/eval_dataset_enzyme.csv": 2,
        "data/eval_dataset_small_molecule.csv": 2,
        "data/eval_dataset_crystal_material.csv": 2,
    }
    n_input = input_dict[data_name]

    return n_input


def data_loader(datasets):
    data_dict = {}

    for data_name in datasets:
        csv_data = []
        with open(data_name, mode='r', newline='') as file:
            csv_data = file.read()

        csv_reader = csv.DictReader(csv_data.splitlines())

        n_input = input_counter(data_name)
        name_matrix = {
            "data/eval_dataset_enzyme.csv": "enzyme",
            "data/eval_dataset_small_molecule.csv": "small_molecule",
            "data/eval_dataset_crystal_material.csv": "crystal_material"
        }
        topic = name_matrix[data_name]
        data_dict[topic] = []
        headers = next(csv_reader)
        headers = list(headers.keys())

        for row in csv_reader:
            content = list(row.values())
            entry = {"input": {}, "label": {}}
            for i in range(n_input):
                entry["input"][headers[i]] = content[i]

            for i in range(n_input, len(headers)):
                entry["label"][headers[i]] = content[i]

            data_dict[topic].append(entry)

    return data_dict


def data_builder(input):
    database = {
        "enzyme": ["data/eval_dataset_enzyme.csv"],
        "small_molecule": ["data/eval_dataset_small_molecule.csv"],
        "crystal_material": ["data/eval_dataset_crystal_material.csv"],
        "All": ["data/eval_dataset_enzyme.csv", "data/eval_dataset_small_molecule.csv", "data/eval_dataset_crystal_material.csv"]
    }
    if input in database.keys():
        datasets = database[input]
    else:
        raise ValueError(f"Unsupported dataset: {input}")
    
    data = data_loader(datasets)

    return data


def example_builder(data, n_examples):
    examples = {}
    new_data = {}

    for key, values in data.items():
        random_selection = random.sample(values, n_examples)
        examples[key] = random_selection
        remaining_values = [v for v in values if v not in random_selection]
        new_data[key] = remaining_values

    return new_data, examples


def test(model, agent, data, examples, n_answers, data_name, prompt_name, output):
    eval_name = "data/eval_dataset.json"
    example_name = "data/example_dataset.json"
    model_name = model[0]["model"]

    with open(eval_name, "w") as json_file:
        json.dump(data, json_file, indent=4)
    with open(example_name, "w") as json_file:
        json.dump(examples, json_file, indent=4)

    ans_list = []
    for n in range(n_answers):
        ans_name = output + f"predict_dataset_{n+1}_{data_name}_{prompt_name}_{model_name}.json"

        prompt = agent(model)
        if len(model) == 2: # GPT based model
            ans = prompt.predict(model, data, examples, ans_name, GPT=True)
        else: # LLaMA based model
            ans = prompt.predict(model, data, examples, ans_name, GPT=False)

        ans_list.append(ans)

        with open(ans_name, "w") as json_file:
            json.dump(ans, json_file, indent=4)

    print("Prediction finished! Start judging!")
    capability = capability_fn(output, data_name, prompt_name, model_name, data, ans_list)
    accuracy, F1_score = accuracy_fn(output, data_name, prompt_name, model_name, data, ans_list)

    return capability, accuracy, F1_score


def infer(input, output, model_config, agent_config, n_examples, n_answers):
    model = Model(model_config)
    agent = TestAgent(agent_config)
    data = data_builder(input)
    left_data, examples = example_builder(data, n_examples)
    print("Data and examples built! Start predicting!")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    capability, accuracy, F1_score = test(model, agent, data, examples, n_answers, input, agent_config, output)
    print("Judgment finished!")