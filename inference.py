import argparse
import json
import logging
import csv
import random
import os

import _jsonnet

from agents.zero_shot import ZeroShotModel
from agents.expert import ExpertModel
from agents.few_shot import FewShotModel
from agents.zero_shot_cot import ZeroShotCoTModel
from agents.few_shot_cot import FewShotCoTModel
from agents.few_shot_cot_critique import FewShotCoTCritiqueModel
from agents.decomposed import DecomposedModel
from eval import capacity_fn, accuracy_fn

logger = logging.getLogger(__name__)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Set configurations for prompt engineering test')
    arg_parser.add_argument('--input', type=str, required=True, help="Input evaluation sets")
    arg_parser.add_argument('--output', type=str, default="results", help="Output file")
    arg_parser.add_argument('--model-config', type=str, default="GPT-3.5", help="LLM model")
    arg_parser.add_argument('--agent-config', type=str, required=True, help="Prompt configs")
    arg_parser.add_argument('--debug', action='store_true', default=False,
                            help="Debug output")
    arg_parser.add_argument('--threads', default=1, type=int,
                            help="Number of threads (use MP if set to >1)")
    arg_parser.add_argument('--n-examples', default=1, type=int,
                            help="Number of few-shot examples (use multi-shot if set to >1)")
    arg_parser.add_argument('--n-answers', default=5, type=int,
                            help="Number of llm generated answers")
    return arg_parser.parse_args()


def Model(model_name):
    if model_name in ["GPT-3.5", "GPT-4"]:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        if api_key is None or api_base is None:
            print("The required environment variables OPENAI_API_KEY or OPENAI_API_BASE have not been set. Please set these environment variables first.")
        api = [api_key, api_base]
        with open('config/gpt_configuration.json', 'r') as config_file:
            all_configs = json.load(config_file)
            config = all_configs[model_name]
            return config, api
    elif model_name in ["LLaMA2-7B", "LLaMA2-13B", "LLaMA2-70B"]:   # 暂且设为同上，具体应该如何config需要等到部署后再讨论
        with open('config/llama_configuration.json', 'r') as config_file:
            all_configs = json.load(config_file)
            config = all_configs[model_name]
            return config
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def TestAgent(agent_name,model_config):
    if agent_name == 'zero-shot':
        return ZeroShotModel(model_config)
    elif agent_name == 'expert':
        return ExpertModel(model_config)
    elif agent_name == 'few-shot':
        return FewShotModel(model_config)
    elif agent_name == 'zero-shot-CoT':
        return ZeroShotCoTModel(model_config)
    elif agent_name == 'few-shot-CoT':
        return FewShotCoTModel(model_config)
    elif agent_name == 'few-shot-CoT-critique':
        return FewShotCoTCritiqueModel(model_config)
    elif agent_name == 'decomposed':
        return DecomposedModel(model_config)
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


def test(model, agent, data, examples, n_answers):
    eval_name = "data/eval_dataset.json"
    example_name = "data/example_dataset.json"

    with open(eval_name, "w") as json_file:
        json.dump(data, json_file)
    with open(example_name, "w") as json_file:
        json.dump(examples, json_file)

    ans_list = []
    for n in range(n_answers):
        if len(model) == 2: # GPT based model
            ans = agent.predict(model, data, examples, n, GPT=True)
        else: # LLaMA based model
            ans = agent.predict(model, data, examples, n, GPT=False)

        ans_list.append(ans)
        ans_name = f"results/predict_dataset_{n+1}.json"

        with open(ans_name, "w") as json_file:
            json.dump(ans, json_file)

    capacity = capacity_fn(ans_list)    # TODO: unfinished
    accuracy = accuracy_fn(data, ans_list)  # TODO: unfinished

    return ans_list, capacity, accuracy


if __name__ == "__main__":

    parsed_args = parse_arguments()
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    model = Model(parsed_args.model_config)
    agent = TestAgent(parsed_args.agent_config, parsed_args.model_config)
    data = data_builder(parsed_args.input)
    left_data, examples = example_builder(data, parsed_args.n_examples)

    ans, capacity, accuracy = test(model, agent, data, examples, parsed_args.n_answers)

    results_dir = parsed_args.output
    # TODO: 肯定不可能直接print，最好是打成表，不过这需要等CoT以及query分类一并完成后再做
    print("Capacity:", capacity)
    print("Accuracy:", accuracy)