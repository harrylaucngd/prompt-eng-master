# prompt base model

import openai
import json
import os
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model):
        if len(model) == 2:
            self.api_key = model[1][0]
            self.api_base = model[1][1]
        else:
            self.api_key = ""
            self.api_base = ""

    def initialize(self, data):
        if isinstance(data, dict):
            return {key: self.initialize(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.initialize(item) for item in data]
        else:
            return None

    @abstractmethod
    def perform_task(self, ans, topic, i, input_name, input_value, label_name, examples, model_name, temp, GPT=True):
        pass

    def alignment(self, model_name, topic, label_name, ans):
        user_msg = [
            {"role": "user", "content": f"For one {topic} and its {label_name}, one gave and answer: {ans}. Please judge if he/she gave a meaningful answer (which means the answer contains exact value or entity or yes/no rather than saying something implicit). If not, return -1."}
        ]
        chat_completion = openai.ChatCompletion.create(
            model=model_name,
            temperature=0.7,
            messages=user_msg
        )
        cap = chat_completion.choices[0].message.content
        user_msg.append({"role": "assistant", "content": cap})

        for i in range(4):
            user_msg.append({"role": "user", "content": "Now examine and simplify the answer, only return the exact value or entity of answer. If content of assistant is -1, only return N/A. If there're multiple answers to multiple questions (including validated and N/A, not for multiple answers to the same question), only take the last group of answers (or single last one answer)."})

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                messages=user_msg
            )
            simplified_ans = chat_completion.choices[0].message.content
            user_msg.append({"role": "assistant", "content": simplified_ans})

        aligned_ans = chat_completion.choices[0].message.content

        return cap, aligned_ans

    def predict(self, model, data, examples, ans_name, GPT=True):
        if GPT:
            model_name = model[0]["model"]
            temp = model[0]["temperature"]
            openai.api_key = self.api_key
            openai.api_base = self.api_base

            # Decompose the dataset into question components and query one by one
            ans = self.initialize(data)

            checkpoint = False   # Skip answers that have already been generated. True: old answer exists; False: no old answer.
            for key in data.keys():
                topic = key
                for i in range(len(data[topic])):
                    entity = data[topic][i]
                    input = entity["input"]
                    input_name = list(input.keys())
                    input_value = list(input.values())
                    labels = entity["label"]
                    for name, value in zip(input_name, input_value):
                        ans[topic][i]["input"][name] = value
                    for label_name in labels.keys():
                        num = len(examples[topic])
                        example = []
                        for k in range(num):
                            example_ensemble = examples[topic][k]
                            example.append({
                                "example_input": [example_ensemble["input"][name] for name in input_name],
                                "example_label": example_ensemble["label"][label_name]
                            })

                        if os.path.exists(ans_name):
                            with open(ans_name, "r") as ans_file:
                                old_ans = json.load(ans_file)
                            if old_ans[topic][i]["label"][label_name]:
                                checkpoint = True
                            else:
                                checkpoint = False
                        else:
                            checkpoint = False

                        if not checkpoint:
                            ans[topic][i]["label"][label_name] = self.perform_task(ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)

                            with open(ans_name, "w") as json_file:
                                json.dump(ans, json_file, indent=4)
                        else:
                            ans = old_ans

            return ans
        else:   # LLaMA inference will be supported later
            raise NotImplementedError