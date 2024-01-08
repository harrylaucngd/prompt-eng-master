import openai
import json
import numpy as np
import scipy.stats as stats
from sklearn.metrics import f1_score
import os
import re
        

def type_judge(topic, label_name):
    cot_classification_name = "data/cot_classification.json"
    with open(cot_classification_name, 'r') as file:
        quest_lists = json.load(file)
    quest_list = quest_lists["All"]

    type = ""
    for key in quest_list.keys():
        if [topic, label_name] in quest_list[key]:
            type = key
            break
    
    return type


def verbal_rating(topic, label_name, data, ans):
    if data != "N/A":
        if "N/A" in ans:
            acc = 0
        else:
            with open('config/api.json', 'r') as api_file:
                all_api = json.load(api_file)
                openai_api = all_api["openai_api"]
            api_key = os.environ.get("OPENAI_API_KEY") or openai_api["api_key"]
            api_base = os.environ.get("OPENAI_API_BASE") or openai_api["api_base"]
            openai.api_key = api_key
            openai.api_base = api_base
            instruction = f"For {topic} {label_name}, one person gave an answer {ans}. The ground truth is {data}. Regardless of whether the unit is written or not, please rate this answer from 0-5. Only return the score as an integer."
            with open('data/eval_prompts.json', 'r') as prompt_file:
                examples = json.load(prompt_file)
            example = examples[topic][label_name]
            instruction += example
            message = [{"role": "user", "content": instruction}]
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                messages=message
            )
            ans = chat_completion.choices[0].message.content
            match = re.search(r"(-?\d+(\.\d+)?)", ans)
            acc = float(match.group(1))
    else:
        acc = 5

    if acc > 5:
        acc = 5
    elif acc <0:
        acc = 0


    return acc/5


def numerical_rating(data, ans):
    neighborhood = {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["C", "E"],
        "E": ["D"]
    }

    if ("N/A" in ans) or ("-1" in ans):
        acc = 0
    else:
        if data in ans:
            acc = 1
        else:
            acc = 0
            neighbors = neighborhood[data]
            for neighbor in neighbors:
                if neighbor in ans:
                    acc = 0.4
    return acc


def choice_determine(ans):
    if ("N/A" in ans) or ("-1" in ans):
        choice = "N/A"
    else:
        choice = "N/A"
        for truth in ["A", "B", "C", "D", "E"]:
            if truth in ans:
                choice = truth
                break
    return choice


def capability_fn(output, data_name, prompt_name, model_name, data, ans_list):
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

    for n in range(len(ans_list)):
        ans = ans_list[n]
        cap_name = output + f"capability_{data_name}_{prompt_name}_{model_name}.json"
        for key in ans.keys():
            topic = key
            for i in range(len(ans[topic])):
                entity = ans[topic][i]
                labels = entity["label"]
                for label_name in labels.keys():
                    if data[topic][i]["label"][label_name] == "N/A":
                        capability[topic][label_name] += 1/(len(ans[topic])*n_answer)
                    elif "N/A" not in ans[topic][i]["label"][label_name]:
                        capability[topic][label_name] += 1/(len(ans[topic])*n_answer)
                    with open(cap_name, "w") as cap_file:
                        json.dump(capability, cap_file, indent=4)
    
    for topic in capability.keys():
        cap_topic = capability[topic]
        for label_name in cap_topic.keys():
            capability[topic][label_name] = round(capability[topic][label_name], 3)
    
    with open(cap_name, "w") as cap_file:
        json.dump(capability, cap_file, indent=4)

    return capability
    

def accuracy_fn(output, data_name, prompt_name, model_name, data, ans_list):
    n_answer = len(ans_list)
    acc_name = output + f"accuracy_{data_name}_{prompt_name}_{model_name}.json"
    if os.path.exists(acc_name):
        with open(acc_name, "r") as json_file:
            accuracy = json.load(json_file)
    else:
        accuracy = {}
        ans = ans_list[0]
        for key in ans.keys():
            topic = key
            accuracy[topic] = {}
            entity = ans[topic][0]
            labels = entity["label"]
            for label_name in labels.keys():
                accuracy[topic][label_name] = 0

    f1_name = output + f"F1_score_{data_name}_{prompt_name}_{model_name}.json"
    if os.path.exists(f1_name):
        with open(f1_name, "r") as json_file:
            F1_score = json.load(json_file)
    else:
        F1_score = {}
        ans = ans_list[0]
        for key in ans.keys():
            topic = key
            F1_score[topic] = {}
            entity = ans[topic][0]
            labels = entity["label"]
            for label_name in labels.keys():
                type = type_judge(topic, label_name)
                if type in ["Numerical & Logical", "Numerical & Experimental"]:
                    F1_score[topic][label_name] = [[], []]  # First list: ans; second list: ground truth

    counterpart = {
        "Substrate": "Product",
        "Product": "Substrate"
    }

    # Verbal & Numerical judging
    i_ans = 0
    for ans in ans_list:
        i_ans += 1
        point_name = output + f"eval_points_{data_name}_{prompt_name}_{model_name}_{i_ans}.txt"
        points = []
        if os.path.exists(point_name):
            with open(point_name, "r") as file:
                for line in file:
                    item = line.strip()
                    points.append(item)
        with open("data/multiple_choices.json", "r") as json_file:
            multiple_choices = json.load(json_file)

        for key in ans.keys():
            topic = key
            for i in range(len(ans[topic])):
                entity = ans[topic][i]
                labels = entity["label"]
                inputs = entity["input"]
                name = next(iter(inputs.values()))
                if name not in points:
                    for label_name in labels.keys():
                        type = type_judge(topic, label_name)
                        if type in ["Verbal & Logical", "Verbal & Experimental"]:
                            if label_name in counterpart.keys():
                                acc1 = float(verbal_rating(topic, label_name, data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name]))
                                counterpart_label = counterpart[label_name]
                                acc2 = float(verbal_rating(topic, label_name, data[topic][i]["label"][counterpart_label], ans[topic][i]["label"][label_name]))
                                acc = max(acc1, acc2)
                            else:
                                acc = float(verbal_rating(topic, label_name, data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name]))
                        elif type in ["Numerical & Logical", "Numerical & Experimental"]:
                            ground_truth = multiple_choices[topic][i]["label"][label_name][1]
                            acc = numerical_rating(ground_truth, ans[topic][i]["label"][label_name])
                            choice = choice_determine(ans[topic][i]["label"][label_name])
                            if choice != "N/A":
                                F1_score[topic][label_name][0].append(choice)
                                F1_score[topic][label_name][1].append(ground_truth)
                        accuracy[topic][label_name] += float(acc)/(len(ans[topic])*n_answer)
                    with open(acc_name, "w") as json_file:
                        json.dump(accuracy, json_file, indent=4)
                    with open(f1_name, "w") as json_file:
                        json.dump(F1_score, json_file, indent=4)
                    points.append(name)
                    with open(point_name, "w") as file:
                        for item in points:
                            file.write(item + "\n")

    for topic in accuracy.keys():
        acc_topic = accuracy[topic]
        for label_name in acc_topic.keys():
            accuracy[topic][label_name] = round(accuracy[topic][label_name], 3)

    with open(acc_name, "w") as acc_file:
        json.dump(accuracy, acc_file, indent=4)
    
    for topic in F1_score.keys():
        F1_topic = F1_score[topic]
        for label_name in F1_topic.keys():
            answer = F1_score[topic][label_name][0]
            ground_truth = F1_score[topic][label_name][1]
            f1_value = f1_score(ground_truth, answer, average='weighted')
            F1_score[topic][label_name] = round(f1_value, 3)

    with open(f1_name, "w") as f1_file:
        json.dump(F1_score, f1_file, indent=4)

    return accuracy, F1_score