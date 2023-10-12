# decomposed prompting (based on ChemCrow)

import openai
import json
from chemcrow.agents import chemcrow

class DecomposedModel:
    def __init__(self, model_config):
        if len(model_config) == 2:
            self.api_key = model_config[1][0]
            self.api_base = model_config[1][1]
        else:
            self.api_key = ""
            self.api_base = ""

    def decomposed(self, topic, input_name, input_value, label_name, model_name, temp, GPT=True):
        if GPT:
            # Define the user message
            user_msg = [
                {"role": "user", "content": f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM:"}
            ]
            chem_model = chemcrow.ChemCrow(
                model=model_name,
                temp=temp,
                max_iterations=2,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                api_keys={}
            )
            response = chem_model.run(user_msg)
            return response
        else:   # LLaMA inference will be supported later
            return "N/A"

    def initialize(self, data):
        if isinstance(data, dict):
            return {key: self.initialize(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.initialize(item) for item in data]
        else:
            return None

    def predict(self, model, data, examples, n, GPT=True):
        if GPT:
            model_name = model[0]["model"]
            temp = model[0]["temperature"]
            api_key = self.api_key
            api_base = self.api_base
            openai.api_key = api_key
            openai.api_base = api_base

            # Decompose the dataset into question components and query one by one
            ans = self.initialize(data)

            for key in data.keys():
                topic = key
                for i in range(len(data[topic])):
                    entity = data[topic][i]
                    input = entity["input"]
                    input_name = input.keys()
                    input_value = input.values()
                    labels = entity["label"]
                    for label_name in labels.keys():
                        for name, value in zip(input_name, input_value):
                            ans[topic][i]["input"][name] = value
                        ans[topic][i]["label"][label_name] = self.decomposed(topic, input_name, input_value, label_name, model_name, temp, GPT=True)
                        
                        ans_name = f"results/predict_dataset_{n+1}.json"

                        with open(ans_name, "w") as json_file:
                            json.dump(ans, json_file)

            return ans
        else:   # LLaMA inference will be supported later
            raise NotImplementedError