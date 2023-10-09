# few-shot-cot prompting

import openai
import json

class FewShotCoTModel:
    def __init__(self, model_config):
        if len(model_config) == 2:
            self.api_key = model_config[1][0]
            self.api_base = model_config[1][1]
        else:
            self.api_key = ""
            self.api_base = ""

    def cot_generation(topic, label_name):
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

        cot_type = ""
        for key in quest_list.keys():
            if [topic, label_name] in quest_list[key]:
                cot_type = key

        cot_example_dataset_name = "data/cot_example_dataset.json"
        with open(cot_example_dataset_name, 'r') as file:
            cot_example_dataset = json.load(file)
        
        cot_example = cot_example_dataset[cot_type][0]  # TODO: Now the dataset is pretty coarse, and it seems not all questions can be solved perfectly with CoT;
                                                        # but we shall discuss and decide how to properly build a support set with delicacy and make it more reasonable

        return cot_example

    def few_shot_cot(self, topic, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        if GPT:
            # Let LLM generate the system message automatically
            instruction = [
                {"role": "system", "content": f"You are an expert in {topic} and are celebrated for your exceptional proficiency in the intricate analysis of {label_name} based on {input_name}."},
                {"role": "user", "content": "Write an expert prompt to persuade me that I am an expert in chemistry too. No longer than 50 words."}
            ]
            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=instruction,
                max_tokens=100
            )

            # Set one round of self asking & self answering to activate the identity of LLM
            messages = []
            system_msg = chat_completion.choices[0].message.content
            user_msg = "Who are you?"

            messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": user_msg})

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=[{"role": "system", "content": system_msg,
                    "role": "user", "content": user_msg}]
            )
            answer = chat_completion.choices[0].message.content
            messages.append({"role": "assistant", "content": answer})

            # Generate the example prompts
            user_msg = ""
            for ex in example:
                ex_input = ex["example_input"]
                ex_label = ex["example_label"]
                user_msg += f"Question: For {topic}, given the {input_name}: {ex_input}, what is the {label_name}?\n LLM: {ex_label}.\n"

            # Add CoT examples
            cot_example = self.cot_generation(topic, label_name)
            user_msg += f"Here is a fake example and let's think step by step: {cot_example}.\n"

            # Define the user message
            user_msg += f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            messages.append({"role": "user", "content": user_msg})

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=messages,
                max_tokens=50
            )
            return chat_completion.choices[0].message.content
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
                        n = len(examples[topic])
                        example = []
                        for k in range(n):
                            example_ensemble = examples[topic][k]
                            example.append({
                                "example_input": [example_ensemble["input"][name] for name in input_name],
                                "example_label": example_ensemble["label"][label_name]
                            })

                        for name, value in zip(input_name, input_value):
                            ans[topic][i]["input"][name] = value
                        ans[topic][i]["label"][label_name] = self.few_shot_cot(topic, input_name, input_value, label_name, example, model_name, temp, GPT=True)

                        ans_name = f"results/predict_dataset_{n+1}.json"

                        with open(ans_name, "w") as json_file:
                            json.dump(ans, json_file)

            return ans
        else:   # LLaMA inference will be supported later
            raise NotImplementedError