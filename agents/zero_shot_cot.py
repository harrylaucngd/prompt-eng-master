# zero-shot-cot prompting

import openai
import json
from agents.base import BaseModel


class ZeroShotCoTModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def zero_shot_cot(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        if GPT:
            # Define the user message
            user_msg = f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM:"
            user_msg += "\nLet's think step by step."
            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=[{"role": "user", "content": user_msg}]
            )
            return chat_completion.choices[0].message.content
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.zero_shot_cot(ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)