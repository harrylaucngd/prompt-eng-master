# few-shot-cot prompting

import openai
import json
import re
from collections import Counter
from agents.base import BaseModel


class FewShotCoTModel(BaseModel):
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
            user_msg.append({"role": "user", "content": "Now examine and simplify the answer, only return the exact value or entity of answer. If content of assistant is -1, only return N/A."})

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                messages=user_msg
            )
            simplified_ans = chat_completion.choices[0].message.content
            user_msg.append({"role": "assistant", "content": simplified_ans})

        aligned_ans = chat_completion.choices[0].message.content

        return cap, aligned_ans

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
        
        cot_example = cot_example_dataset[topic][label_name]

        return quest_lists, cot_example
    
    def pick(self, topic, label_name, answers):
        cot_classification_name = "data/cot_classification.json"
        with open(cot_classification_name, 'r') as file:
            quest_lists = json.load(file)
        quest_list = quest_lists["All"]
        if ([topic, label_name] in quest_list["Numerical & Logical"]) or ([topic, label_name] in quest_list["Numerical & Experimental"]):
            numbers = []
            for answer in answers:
                if "N/A" in answer:
                    continue
                else:
                    match = re.search(r"(-?\d+(\.\d+)?)", answer)
                    number = match.group(1)
                    numbers.append(float(number))
            if numbers != []:
                assembled_answer = sum(numbers) / len(numbers)
                if "Number" in label_name:
                    assembled_answer = round(assembled_answer)
                else:
                    assembled_answer = round(assembled_answer, 3)
            else:
                assembled_answer = "N/A"
        else:
            verbals = []
            for answer in answers:
                if "N/A" in answer:
                    continue
                else:
                    verbals.append(answer)
            if verbals != []:
                count = Counter(verbals)
                assembled_answer = count.most_common(1)[0][0]
            else:
                assembled_answer = "N/A"
        return assembled_answer

    def few_shot_cot(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
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
            msg = user_msg + f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            msgs = messages.copy()
            msgs.append({"role": "user", "content": msg})

            answers = []
            for num in range(3):
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
            if [topic, label_name] in quest_lists["small_molecule"]["Logical"]:
                molecular_formula = ans[topic][i]["label"]["Molecular Formula"]
                smiles = ans[topic][i]["input"]["SMILES"]
                user_msg += f"Now knowing the Molecular Formula: {molecular_formula} and Smiles: {smiles}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            elif [topic, label_name] in quest_lists["small_molecule"]["Comprehensive"]:
                molecular_weight = ans[topic][i]["label"]["Molecular Weight (unit: g/mol)"]
                solubility = ans[topic][i]["label"]["Solubility (in water, unit: mg/L)"]
                hba = ans[topic][i]["label"]["Number of H-bond Acceptors"]
                hbd = ans[topic][i]["label"]["Number of H-bond Donors"]
                logp = ans[topic][i]["label"]["LogP"]
                user_msg += f"Now knowing the Molecular Weight (unit: g/mol): {molecular_weight}, Solubility (in water, unit: mg/L): {solubility}, Number of H-bond Acceptors: {hba}, Number of H-bond Donors: {hbd} and LogP: {logp}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:   # TODO: We shall complete all detailed cot design later.
                user_msg += f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            messages.append({"role": "user", "content": user_msg})

            for num in range(3):
                chat_completion = openai.ChatCompletion.create(
                    model=model_name,
                    temperature=temp,
                    messages=messages
                )
                answers.append(chat_completion.choices[0].message.content)

            for i, answer in enumerate(answers):
                cap, aligned_answer = self.alignment(model_name, topic, label_name, answer)
                if "-1" not in cap:
                    answers[i] = aligned_answer
                else:
                    answers[i] = "N/A"

            answer = self.pick(topic, label_name, answers)
            return answer
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.few_shot_cot(ans, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)