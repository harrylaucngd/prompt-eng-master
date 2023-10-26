# zero-shot prompting

import openai
import json
from agents.base import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

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
            user_msg.append({"role": "user", "content": "Now examine and simplify the answer, only return the exact value or entity of answer. If content of assistant is -1, only return N/A. If there're multiple answers (including validated and N/A), only take the last one."})

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                messages=user_msg
            )
            simplified_ans = chat_completion.choices[0].message.content
            user_msg.append({"role": "assistant", "content": simplified_ans})

        aligned_ans = chat_completion.choices[0].message.content

        return cap, aligned_ans

    def zero_shot(self, topic, input_name, input_value, label_name, model_name, temp, GPT=True):
        if GPT:
            # Define the user message
            user_msg = [
                {"role": "user", "content": f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM:"}
            ]
            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temp,
                messages=user_msg
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
        return self.zero_shot(topic, input_name, input_value, label_name, model_name, temp, GPT=True)