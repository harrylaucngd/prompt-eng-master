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


def alignment(topic, label_name, ans):
    user_msg = [
        {"role": "user", "content": f"For one {topic} and its {label_name}, one gave and answer: {ans}. Please judge if he/she gave a meaningful answer (which means the answer contains exact value or entity or yes/no rather than saying something implicit). If not, return -1."}
    ]
    chat_completion = openai.ChatCompletion.create(
        engine="GPT-3.5-turbo",
        temperature=0.7,
        messages=user_msg
    )
    cap = chat_completion.choices[0].message.content
    user_msg.append({"role": "assistant", "content": cap})

    for i in range(4):
        user_msg.append({"role": "user", "content": "Now examine and simplify the answer, only return the exact value or entity of answer. If content of assistant is -1, only return N/A."})

        chat_completion = openai.ChatCompletion.create(
            engine="GPT-3.5-turbo",
            temperature=0.7,
            messages=user_msg
        )
        simplified_ans = chat_completion.choices[0].message.content
        user_msg.append({"role": "assistant", "content": simplified_ans})

    aligned_ans = chat_completion.choices[0].message.content

    return cap, aligned_ans


def verbal_rating(topic, label_name, data, ans):
    # TODO
    acc = 0.5
    return acc


def numerical_rating(data, ans):
    # TODO
    acc = 0.5
    return acc


def capability_fn(output, data_name, prompt_name, model_name, ans_list):
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

    for n in range(len(ans_list)):
        ans = ans_list[n]
        aligned_ans = initialize(ans)
        cap_name = output + f"capability_{data_name}_{prompt_name}_{model_name}.json"
        ans_name = output + f"predict_dataset_{n+1}_{data_name}_{prompt_name}_{model_name}_aligned.json"
        checkpoint = False   # Skip answers that have already been generated. True: old answer exists; False: no old answer.
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
                    if os.path.exists(ans_name):
                        with open(ans_name, "r") as ans_file:
                            old_aligned_ans = json.load(ans_file)
                        if old_aligned_ans[topic][i]["label"][label_name]:
                            checkpoint = True
                        else:
                            checkpoint = False
                    else:
                        checkpoint = False

                    if not checkpoint:
                        cap, aligned_ans[topic][i]["label"][label_name] = alignment(topic, label_name, ans[topic][i]["label"][label_name])
                        if "-1" not in cap:
                            capability[topic][label_name] += 1/(len(ans[topic])*n_answer)
                        else:
                            aligned_ans[topic][i]["label"][label_name] = "N/A"
                        with open(ans_name, "w") as ans_file:
                            json.dump(aligned_ans, ans_file, indent=4)
                        with open(cap_name, "w") as cap_file:
                            json.dump(capability, cap_file, indent=4)
                    else:
                        aligned_ans = old_aligned_ans

        aligned_ans_list.append(aligned_ans)

    return aligned_ans_list, capability
    

def accuracy_fn(output, data_name, prompt_name, model_name, data, ans_list):
    set_env()
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
    
    for ans in ans_list:
        acc_name = output + f"capability_{data_name}_{prompt_name}_{model_name}.json"
        for key in ans.keys():
            topic = key
            for i in range(len(ans[topic])):
                entity = ans[topic][i]
                labels = entity["label"]
                for label_name in labels.keys():
                    type = type_judge(topic, label_name)
                    if type in ["", ""]:
                        acc = float(verbal_rating(topic, label_name, data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name]))/5
                    else:
                        acc = numerical_rating(topic, label_name, data[topic][i]["label"][label_name], ans[topic][i]["label"][label_name])
                    accuracy[topic][label_name] += float(acc)/(len(ans[topic])*n_answer)
                    with open(acc_name, "w") as json_file:
                        json.dump(ans, json_file, indent=4)

    return accuracy