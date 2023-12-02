# few-shot prompting

import openai
import json
from eval import type_judge
from agents.base import BaseModel


class FewShotModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def few_shot(self, ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
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

            # Define the user message
            quest_lists, cot_example = self.cot_generation(topic, label_name)
            with open("data/multiple_choices.json", "r") as json_file:
                multiple_choices = json.load(json_file)
            choices = multiple_choices[topic][i]["label"][label_name][0]
            user_msg = self.Non_CoT_query(self, user_msg, data, topic, i, label_name, quest_lists, input_name, input_value, choices)
            messages.append({"role": "user", "content": user_msg})

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=messages
            )
            answer = chat_completion.choices[0].message.content
            cap, aligned_answer = self.alignment(model_name, topic, label_name, answer)
            if "-1" not in cap:
                answer = aligned_answer
            else:
                answer = "N/A"
            return answer
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.few_shot_cot(ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)