# zero-shot prompting

import openai
import json
from eval import type_judge
from agents.base import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def zero_shot(self, data, topic, input_name, input_value, label_name, i, model_name, temp, GPT=True):
        if GPT:
            # Define the user message
            quest_lists, cot_example = self.cot_generation(topic, label_name)
            with open("data/multiple_choices.json", "r") as json_file:
                multiple_choices = json.load(json_file)
            type = type_judge(topic, label_name)
            if type in ["Verbal & Logical", "Verbal & Experimental"]:
                choices = ""
            else:
                choices = multiple_choices[topic][i]["label"][label_name][0]
            message = ""
            message = self.Non_CoT_query(message, data, topic, i, label_name, quest_lists, input_name, input_value, choices)
            user_msg = [
                {"role": "user", "content": message}
            ]
            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=user_msg
            )
            answer = chat_completion.choices[0].message.content
            cap, aligned_answer = self.alignment(model_name, topic, label_name, answer, choices)
            if "-1" not in cap:
                answer = aligned_answer
            else:
                answer = "N/A"
            return answer
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.zero_shot(data, topic, input_name, input_value, label_name, i, model_name, temp, GPT=True)