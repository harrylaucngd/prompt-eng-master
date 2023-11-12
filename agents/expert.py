# expert prompting

import openai
import json
from eval import type_judge
from agents.base import BaseModel


class ExpertModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def expert(self, topic, input_name, input_value, label_name, model_name, temp, GPT=True):
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

            # Define the user message
            user_msg = f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            type = type_judge(topic, label_name)
            if type == "Binary":
                user_msg += "\nPlease notice that you should return a number from 0 to 10 as a reply."
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

    def perform_task(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.expert(topic, input_name, input_value, label_name, model_name, temp, GPT=True)