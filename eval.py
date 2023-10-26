import openai
import json
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
    with open('config/api.json', 'r') as api_file:
        all_api = json.load(api_file)
        openai_api = all_api["openai_api"]
    api_key = os.environ.get("OPENAI_API_KEY") or openai_api["api_key"]
    api_base = os.environ.get("OPENAI_API_BASE") or openai_api["api_base"]
    openai.api_key = api_key
    openai.api_base = api_base
    instruction = f"For {topic} {label_name}, one person gave an answer {ans}. The ground truth is {data}. Regardless of whether the unit is written or not, please rate this answer from 0-5. Only return the score."
    message = [{"role": "user", "content": instruction}]
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        messages=message
    )
    ans = chat_completion.choices[0].message.content
    match = re.search(r"(-?\d+(\.\d+)?)", ans)
    acc = float(match.group(1))
    return acc/5


def numerical_rating(data, ans):
    if ans == "N/A":
        acc = 0
    else:
        try:
            match1 = re.search(r"(-?\d+(\.\d+)?)", data)
            data_num = float(match1.group(1))
        except:
            data_num = float(data)
        try:
            match2 = re.search(r"(-?\d+(\.\d+)?)", ans)
            ans_num = float(match2.group(1))
        except:
            ans_num = float(ans)
        acc = 1 - abs((ans_num-data_num)/data_num)
    return acc


def capability_fn(output, data_name, prompt_name, model_name, ans_list):
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
                    if ans[topic][i]["label"][label_name] != "N/A":
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
    acc_name = output + f"accuracy_{data_name}_{prompt_name}_{model_name}.json"
    if os.path.exists(acc_name):
        with open(acc_name, "r") as json_file:
            accuracy = json.load(json_file)
    else:
        accuracy = {}
        n_answer = len(ans_list)

        ans = ans_list[0]
        for key in ans.keys():
            topic = key
            accuracy[topic] = {}
            entity = ans[topic][0]
            labels = entity["label"]
            for label_name in labels.keys():
                accuracy[topic][label_name] = 0
    
    point_name = output + f"eval_points_{data_name}_{prompt_name}_{model_name}.txt"
    points = []
    if os.path.exists(point_name):
        with open(point_name, "r") as file:
            for line in file:
                item = line.strip()
                points.append(item)

    for ans in ans_list:
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
                            acc = float(verbal_rating(topic, label_name, data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name]))
                        else:
                            acc = numerical_rating(data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name])
                        accuracy[topic][label_name] += float(acc)/(len(ans[topic])*n_answer)
                    with open(acc_name, "w") as json_file:
                        json.dump(accuracy, json_file, indent=4)
                    points.append(name)
                    with open(point_name, "w") as file:
                        for item in points:
                            file.write(item + "\n")

    for topic in accuracy.keys():
        cap_topic = accuracy[topic]
        for label_name in cap_topic.keys():
            accuracy[topic][label_name] = round(accuracy[topic][label_name], 3)
    
    with open(acc_name, "w") as cap_file:
        json.dump(accuracy, cap_file, indent=4)

    return accuracy