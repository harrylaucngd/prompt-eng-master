import openai
import json
import os


def set_env():
    with open('config/api.json', 'r') as api_file:
        all_api = json.load(api_file)
        openai_api = all_api["openai_api"]
    api_key = os.environ.get("OPENAI_API_KEY") or openai_api["api_key"]
    api_base = os.environ.get("OPENAI_API_BASE") or openai_api["api_base"]
    if api_key is None or api_base is None:
        print("The required environment variables OPENAI_API_KEY or OPENAI_API_BASE have not been set. Please set these environment variables first.")
    else:
        openai.api_key = api_key
        openai.api_base = api_base


def initialize(data):
        if isinstance(data, dict):
            return {key: initialize(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [initialize(item) for item in data]
        else:
            return None
        

def type_judge(topic, label_name):
    quest_list = {
        "Numerical & Logical": [
            ["enzyme", "Substrate"],
            ["enzyme", "Product"],
            ["small_molecule", "Molecular weight"],
            ["small_molecule", "H-bond donors"],
            ["small_molecule", "H-bond acceptors"]
        ],
        "Verbal & Logical": [
            ["small_molecule", "Molecular formula"],
            ["crystal_material", "Ground State Phase"]
        ],
        "Numerical & Experimental": [
            ["enzyme", "Km"],
            ["small_molecule", "Boiling Point"],
            ["small_molecule", "Density"],
            ["small_molecule", "logP"],
            ["small_molecule", "tPSA"],
            ["small_molecule", "Apolar desolvation (kcal/mol)"],
            ["small_molecule", "Polar desolvation (kcal/mol)"],
            ["crystal_material", "ΔfH"],
            ["crystal_material", "Decomposition Energy"]
        ],
        "Verbal & Experimental": [
            ["enzyme", "Active site"],
            ["crystal_material", "Competing Phases"]
        ]
    }

    type = ""
    for key in quest_list.keys():
        if [topic, label_name] in quest_list[key]:
            type = key
            break
    
    return type


def alignment(topic, label_name, ans):
    user_msg = [
        {"role": "user", "content": f"{topic}{label_name}{ans}"}
    ]
    chat_completion = openai.ChatCompletion.create(
        model="GPT-4",
        temperature=0.1,
        messages=user_msg
    )
    cap = chat_completion.choices[0].message.content
    user_msg.append({"role": "assistant", "content": cap})

    for i in range(4):
        user_msg.append({"role": "user", "content": "Now examine and simplify the answer, only return the exact value or entity of answer. If content of assistant is -1, only return N/A."})

        chat_completion = openai.ChatCompletion.create(
            model="GPT-4",
            temperature=0.1,
            messages=user_msg
        )
        simplified_ans = chat_completion.choices[0].message.content
        user_msg.append({"role": "assistant", "content": simplified_ans})

    aligned_ans = chat_completion.choices[0].message.content

    return cap, aligned_ans


def capability_fn(ans_list):
    set_env()
    aligned_ans_list = []
    capability = {}
    n_answer = len(ans_list)

    ans = ans_list[0]
    for key in ans.keys():
         topic = key
         capability[topic] = {}
         entity = ans[topic][0]
         labels = entity["label"]
         for label_name in labels.keys():
              capability[topic][label_name] = 0

    for ans in ans_list:
        aligned_ans = initialize(ans)
        for key in ans.keys():
            topic = key
            for i in range(len(ans[topic])):
                entity = ans[topic][i]
                input = entity["input"]
                input_name = list(input.keys())
                input_value = list(input.values())
                labels = entity["label"]
                for name, value in zip(input_name, input_value):
                    aligned_ans[topic][i]["input"][name] = value
                for label_name in labels.keys():
                    cap, aligned_ans[topic][i]["label"][label_name] = alignment(topic, label_name, ans[topic][i]["label"][label_name])
                    if "-1" in cap:
                        capability[topic][label_name] += 1/(len(ans[topic])*n_answer)
                    else:
                        aligned_ans[topic][i]["label"][label_name] = "N/A"

    aligned_ans_list.append(aligned_ans)

    return aligned_ans_list, capability
    

def accuracy_fn(data, ans_list):
    set_env()
    # TODO
    accuracy = 0.5

    return accuracy