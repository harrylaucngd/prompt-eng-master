# zero-shot-cot prompting

import openai
import json
from eval import type_judge
from agents.base import BaseModel


class ZeroShotCoTModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def zero_shot_cot(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        if GPT:
            # Define the user message
            user_msg = f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM:"
            type = type_judge(topic, label_name)
            if type == "Binary":
                user_msg += "\nPlease notice that you should return a number from 0 to 10 as a reply."
            user_msg += "\nLet's think step by step."
            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=[{"role": "user", "content": user_msg}]
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

    def perform_task(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.zero_shot_cot(ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)