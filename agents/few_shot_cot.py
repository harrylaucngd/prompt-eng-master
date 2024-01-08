# few-shot-cot prompting

import openai
import json
import re
from eval import type_judge
from collections import Counter
from agents.base import BaseModel


class FewShotCoTModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def pick(self, label_name, answers):
        realans = []
        if "Yes or No" in label_name:
            for answer in answers:
                if ("yes" in answer) or ("Yes" in answer) or ("YES" in answer):
                    answer = "Yes"
                elif ("no" in answer) or ("No" in answer) or ("NO" in answer):
                    answer = "No"
                else:
                    answer = "N/A"
        for answer in answers:
            if "N/A" in answer:
                continue
            else:
                realans.append(answer)
        if realans != []:
            count = Counter(realans)
            assembled_answer = count.most_common(1)[0][0]
        else:
            assembled_answer = "N/A"

        return assembled_answer

    def few_shot_cot(self, ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        if GPT:
            expert_prompts = [
                f"You are an expert in {topic} and are celebrated for your exceptional proficiency in the intricate analysis of {label_name} based on {input_name}.",
                f"Your reputation as a master in {topic} precedes you, especially due to your innovative methods in interpreting {label_name} derived from {input_name}. Your insights have revolutionized the understanding of this domain.",
                f"As a distinguished expert in {topic}, your skill in conducting detailed examinations of {label_name} using {input_name} is unparalleled. Your work has not only advanced the field but also provided new perspectives on critical issues."
            ]

            answers = []
            for i in range(3):  # 3 Mix experts CoT assembly
                # Let LLM generate the system message automatically
                instruction = [
                    {"role": "system", "content": expert_prompts[i]},
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
                msg = user_msg + f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
                msgs = messages.copy()
                msgs.append({"role": "user", "content": msg})

                chat_completion = openai.ChatCompletion.create(
                    model=model_name,
                    temperature=temp,
                    messages=msgs
                )
                answers.append(chat_completion.choices[0].message.content)

                # Add CoT examples
                quest_lists, cot_example = self.cot_generation(topic, label_name)
                user_msg += f"Here is a fake example and let's think step by step: {cot_example}.\n"

                # Define the user message
                with open("data/multiple_choices.json", "r") as json_file:
                    multiple_choices = json.load(json_file)
                type = type_judge(topic, label_name)
                if type in ["Verbal & Logical", "Verbal & Experimental"]:
                    choices = ""
                else:
                    choices = multiple_choices[topic][i]["label"][label_name][0]
                user_msg = self.CoT_query(user_msg, data, topic, i, label_name, quest_lists, input_name, input_value, choices)
                messages.append({"role": "user", "content": user_msg})

                chat_completion = openai.ChatCompletion.create(
                    model=model_name,
                    temperature=temp,
                    messages=messages
                )
                answers.append(chat_completion.choices[0].message.content)

            for i, answer in enumerate(answers):    # Assembling answers
                cap, aligned_answer = self.alignment(model_name, topic, label_name, answer, choices)
                if "-1" not in cap:
                    answers[i] = aligned_answer
                else:
                    answers[i] = "N/A"
            answer = self.pick(label_name, answers)

            return answer
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.few_shot_cot(ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)