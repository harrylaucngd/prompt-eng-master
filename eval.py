import openai
import json
import scipy.stats as stats
import numpy as np
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

    return acc/5


def find_stddev(a):
    lower, upper = 0, 2*abs(a)
    tol = 1e-6
    while upper-lower > tol:
        mid = (lower + upper) / 2
        cdf_lower = stats.norm.cdf(a - abs(a), loc=a, scale=mid)
        cdf_upper = stats.norm.cdf(a + abs(a), loc=a, scale=mid)
        probability = cdf_upper - cdf_lower
        if probability < 0.5:
            upper = mid
        else:
            lower = mid
    return (lower + upper) / 2


def calculate_score(a, b):
    if a != 0:
        stddev = find_stddev(a)
    else:
        stddev = 1
    cdf_value = stats.norm.cdf(b, loc=a, scale=stddev)
    score = 1 - abs(cdf_value - 0.5) * 2
    return score


def numerical_rating(data, ans):
    if ans == "N/A":
        acc = 0
    else:
        try:
            match1 = re.search(r"(-?\d+(\.\d+)?)", data)
            data_num = float(match1.group(1))
        except:
            try:
                data_num = float(data)
            except:
                with open("data/number_transform.json", "r") as json_file:
                    transform = json.load(json_file)
                data_num = 0
                for num in transform.keys():
                    if num in data:
                        data_num = transform[num]
                        break
        try:
            match2 = re.search(r"(-?\d+(\.\d+)?)", ans)
            ans_num = float(match2.group(1))
        except:
            try:
                ans_num = float(ans)
            except:
                with open("data/number_transform.json", "r") as json_file:
                    transform = json.load(json_file)
                ans_num = 0
                for num in transform.keys():
                    if num in ans:
                        ans_num = transform[num]
                        break
        acc = calculate_score(data_num, ans_num)
    return acc


def align(data):
    data_num = 5
    if data != "N/A":
        try:
            match = re.search(r"(-?\d+(\.\d+)?)", data)
            data_num = float(match.group(1))
        except:
            try:
                data_num = float(data)
            except:
                with open("data/number_transform.json", "r") as json_file:
                    transform = json.load(json_file)
                for num in transform.keys():
                    if num in data:
                        data_num = transform[num]
                        break

    return data_num

def softmax(ans_dict, data_dict):
    acc_dict = {}
    if ans_dict != {}:
        for label_name in ans_dict.keys():
            ans_array = np.array(ans_dict[label_name])
            data_list = data_dict[label_name]

            ans_array -= np.max(ans_array)
            p = np.exp(ans_array) / np.sum(ans_array)
            ranking = np.argsort(p)[::-1]  # Sort in descending order
            n = len(ranking)

            threshold = n // 2
            result_labels = ['Yes' if i < threshold else 'No' for i in range(n)]
            total_score = sum([1 if label == data_list[idx] else 0 for idx, label in enumerate(result_labels)])

            acc_dict[label_name] = total_score

    return acc_dict


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
                    elif ans[topic][i]["label"][label_name] != "N/A":
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
                            acc = numerical_rating(data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name])
                        accuracy[topic][label_name] += float(acc)/(len(ans[topic])*n_answer)
                    with open(acc_name, "w") as json_file:
                        json.dump(accuracy, json_file, indent=4)
                    points.append(name)
                    with open(point_name, "w") as file:
                        for item in points:
                            file.write(item + "\n")

    # Binary softmax judging
    for ans in ans_list:
        ans_dict, data_dict = {}, {}
        for topic in accuracy.keys():
            entity = accuracy[topic]
            for label_name in entity.keys():
                type = type_judge(topic, label_name)
                if type == "Binary":
                    ans_dict[label_name] = []
                    data_dict[label_name] = []

        for key in ans.keys():
            topic = key
            for i in range(len(ans[topic])):
                entity = ans[topic][i]
                labels = entity["label"]
                for label_name in labels.keys():
                    type = type_judge(topic, label_name)
                    if type == "Binary":
                        answer = align(ans[topic][i]["label"][label_name])
                        ground_truth = data[topic][i]["label"][label_name]
                        ans_dict[label_name].append(answer)
                        data_dict[label_name].append(ground_truth)

        acc_dict = softmax(ans_dict, data_dict)
        for label_name in acc_dict.keys():
            for topic in accuracy.keys():
                if label_name in accuracy[topic].keys():
                    accuracy[topic][label_name] += acc_dict[label_name] / len(ans_list)
                    break

    for topic in accuracy.keys():
        acc_topic = accuracy[topic]
        for label_name in acc_topic.keys():
            accuracy[topic][label_name] = round(accuracy[topic][label_name], 3)
    
    with open(acc_name, "w") as acc_file:
        json.dump(accuracy, acc_file, indent=4)

    return accuracy