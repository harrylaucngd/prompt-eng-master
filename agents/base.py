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
    def perform_task(self, topic, input_name, input_value, label_name, examples, model_name, temp, GPT=True):
        pass

    def predict(self, model, data, examples, ans_name, GPT=True):
        if GPT:
            model_name = model[0]["model"]
            temp = model[0]["temperature"]
            openai.api_key = self.api_key
            openai.api_base = self.api_base

            # Decompose the dataset into question components and query one by one
            ans = self.initialize(data)

            for key in data.keys():
                checkpoint1 = True   # skip answers that have already been generated
                checkpoint2 = True
                topic = key
                for i in range(len(data[topic])):
                    entity = data[topic][i]
                    input = entity["input"]
                    input_name = input.keys()
                    input_value = input.values()
                    labels = entity["label"]
                    for label_name in labels.keys():
                        num = len(examples[topic])
                        example = []
                        for k in range(num):
                            example_ensemble = examples[topic][k]
                            example.append({
                                "example_input": [example_ensemble["input"][name] for name in input_name],
                                "example_label": example_ensemble["label"][label_name]
                            })

                        if checkpoint2 & checkpoint1:
                            if os.path.exists(ans_name):
                                with open(ans_name, "r") as ans_file:
                                    old_ans = json.load(ans_file)
                                if old_ans[topic][i]["label"][label_name]:
                                    checkpoint1 = False
                            else:
                                checkpoint2 = False

                        if checkpoint1:
                            for name, value in zip(input_name, input_value):
                                ans[topic][i]["input"][name] = value
                            ans[topic][i]["label"][label_name] = self.perform_task(topic, input_name, input_value, label_name, example, model_name, temp, GPT=True)

                            with open(ans_name, "w") as json_file:
                                json.dump(ans, json_file, indent=4)
                        else:
                            ans = old_ans

            return ans
        else:   # LLaMA inference will be supported later
            raise NotImplementedError