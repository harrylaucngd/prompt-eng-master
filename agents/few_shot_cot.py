# few-shot-cot prompting

import openai
import json
from agents.base import BaseModel


class FewShotCoTModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def cot_generation(self, topic, label_name):
        cot_classification_name = "data/cot_classification.json"
        with open(cot_classification_name, 'r') as file:
            quest_lists = json.load(file)
        quest_list = quest_lists[topic]

        cot_type = ""
        for key in quest_list.keys():
            if [topic, label_name] in quest_list[key]:
                cot_type = key
                break

        cot_example_dataset_name = "data/cot_example_dataset.json"
        with open(cot_example_dataset_name, 'r') as file:
            cot_example_dataset = json.load(file)
        
        cot_example = cot_example_dataset[topic][cot_type][0]  # TODO: Now the dataset is pretty coarse, and it seems not all questions can be solved perfectly with CoT;
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
                messages=instruction
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
                messages=messages
            )
            return chat_completion.choices[0].message.content
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, topic, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.few_shot_cot(topic, input_name, input_value, label_name, example, model_name, temp, GPT=True)