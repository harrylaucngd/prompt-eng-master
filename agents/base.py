# prompt base model

import openai
import json
import os
from abc import ABC, abstractmethod
from eval import type_judge


class BaseModel(ABC):
    def __init__(self, model):
        if len(model) == 2:
            self.api_key = model[1][0]
            self.api_base = model[1][1]
        else:
            self.api_key = ""
            self.api_base = ""

    def initialize(self, data):
        if isinstance(data, dict):
            return {key: self.initialize(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.initialize(item) for item in data]
        else:
            return None
        
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

    @abstractmethod
    def perform_task(self, ans, data, topic, i, input_name, input_value, label_name, examples, model_name, temp, GPT=True):
        pass

    def alignment(self, model_name, topic, label_name, ans, choices):    
        user_msg = [
            {"role": "user", "content": f"For one {topic} and its {label_name}, one gave and answer: {ans}. Please judge if he/she gave a meaningful answer (which means the answer contains exact value or entity or choices like A/B/C/D/E or yes/no rather than saying something implicit). If not, return -1."}
        ]    
        chat_completion = openai.ChatCompletion.create(
            model=model_name,
            temperature=0.7,
            messages=user_msg
        )
        cap = chat_completion.choices[0].message.content
        user_msg.append({"role": "assistant", "content": cap})

        type = type_judge(topic, label_name)
        if type in ["Verbal & Logical", "Verbal & Experimental"]:
            msg = {"role": "user", "content": "Now examine and simplify the answer, only return the exact numerical value, entity, multiple choice options or yes/no of answer. If content of assistant is -1, only return N/A. If there're multiple answers to multiple questions (including validated and N/A, not for multiple answers to the same question), only take the last group of answers (or single last one answer)."}
        else:
            msg = {"role": "user", "content": f"Try simplify and only return A/B/C/D/E but nothing else. Now examine and simplify the answer. If one provides an answer including choices like A/B/C/D/E, only return A/B/C/D/E (one single letter) and nothing else. If one gives the answer as a number or interval, here’s the original multiple choices: {choices}, please manually check which choice it corresponds to, and only return A/B/C/D/E (one single letter) and nothing else. If not match, return N/A."}

        for i in range(4):
            user_msg.append(msg)

            chat_completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                messages=user_msg
            )
            simplified_ans = chat_completion.choices[0].message.content
            user_msg.append({"role": "assistant", "content": simplified_ans})

        aligned_ans = chat_completion.choices[0].message.content

        return cap, aligned_ans
    
    def Non_CoT_query(self, user_msg, data, topic, i, label_name, quest_lists, input_name, input_value, choices):
        if [topic, label_name] in quest_lists["small_molecule"]["Logical"]:
            molecular_formula = data[topic][i]["label"]["Molecular Formula"]
            smiles = data[topic][i]["input"]["SMILES"]
            if choices == "":
                user_msg += f"Question: Now knowing the Molecular Formula: {molecular_formula} and Smiles: {smiles}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the Molecular Formula: {molecular_formula} and Smiles: {smiles}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["small_molecule"]["Comprehensive"]:
            molecular_weight = data[topic][i]["label"]["Molecular Weight (unit: g/mol)"]
            solubility = data[topic][i]["label"]["Solubility (in water, unit: mg/L)"]
            hba = data[topic][i]["label"]["Number of H-bond Acceptors"]
            hbd = data[topic][i]["label"]["Number of H-bond Donors"]
            logp = data[topic][i]["label"]["LogP"]
            if choices == "":
                user_msg += f"Question: Now knowing the Molecular Weight (unit: g/mol): {molecular_weight}, Solubility (in water, unit: mg/L): {solubility}, Number of H-bond Acceptors: {hba}, Number of H-bond Donors: {hbd} and LogP: {logp}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the Molecular Weight (unit: g/mol): {molecular_weight}, Solubility (in water, unit: mg/L): {solubility}, Number of H-bond Acceptors: {hba}, Number of H-bond Donors: {hbd} and LogP: {logp}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Stability"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            energy_above_hull = data[topic][i]["label"]["Energy Above Hull (unit: eV/atom)"]
            if choices == "":
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Energy Above Hull (unit: eV/atom): {energy_above_hull}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Energy Above Hull (unit: eV/atom): {energy_above_hull}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Vector"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            lattice_angle = [data[topic][i]["label"]["Lattice Angle α (among 3 angles as [α, β, γ])"], data[topic][i]["label"]["Lattice Angle β (among 3 angles as [α, β, γ])"], data[topic][i]["label"]["Lattice Angle γ (among 3 angles as [α, β, γ])"]]
            if choices == "":
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Angle [α, β, γ]: {lattice_angle}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Angle [α, β, γ]: {lattice_angle}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Metal"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            band_gap = data[topic][i]["label"]["Band Gap (unit: eV)"]
            if choices == "":
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Band Gap (unit: eV): {band_gap}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Band Gap (unit: eV): {band_gap}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Ordering"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            total_magnetization = data[topic][i]["label"]["Total Magnetization (unit: µB/f.u.)"]
            if choices == "":
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Total Magnetization (unit: µB/f.u.): {total_magnetization}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Total Magnetization (unit: µB/f.u.): {total_magnetization}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Density"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            lattice_vector = [data[topic][i]["label"]["a in Lattice Vector [a, b, c] (unit: Å)"], data[topic][i]["label"]["b in Lattice Vector [a, b, c] (unit: Å)"], data[topic][i]["label"]["c in Lattice Vector [a, b, c] (unit: Å)"]]
            if choices == "":
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Vector [a, b, c] (unit: Å): {lattice_vector}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Vector [a, b, c] (unit: Å): {lattice_vector}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["enzyme"]["Comprehensive"]:
            enzyme = data[topic][i]["input"]["Enzyme"]
            substrate = data[topic][i]["label"]["Substrate"]
            product = data[topic][i]["label"]["Product"]
            if choices == "":
                user_msg += f"Question: Now knowing the Enzyme: {enzyme}, Substrate: {substrate} and Product: {product}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: Now knowing the Enzyme: {enzyme}, Substrate: {substrate} and Product: {product}, for {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        else:
            if choices == "":
                user_msg += f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Consider the following: {choices}\n LLM: "
        return user_msg
    
    def CoT_query(self, user_msg, data, topic, i, label_name, quest_lists, input_name, input_value, choices):
        if [topic, label_name] in quest_lists["small_molecule"]["Logical"]:
            molecular_formula = data[topic][i]["label"]["Molecular Formula"]
            smiles = data[topic][i]["input"]["SMILES"]
            if choices == "":
                user_msg += f"Now knowing the Molecular Formula: {molecular_formula} and Smiles: {smiles}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the Molecular Formula: {molecular_formula} and Smiles: {smiles}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["small_molecule"]["Comprehensive"]:
            molecular_weight = data[topic][i]["label"]["Molecular Weight (unit: g/mol)"]
            solubility = data[topic][i]["label"]["Solubility (in water, unit: mg/L)"]
            hba = data[topic][i]["label"]["Number of H-bond Acceptors"]
            hbd = data[topic][i]["label"]["Number of H-bond Donors"]
            logp = data[topic][i]["label"]["LogP"]
            if choices == "":
                user_msg += f"Now knowing the Molecular Weight (unit: g/mol): {molecular_weight}, Solubility (in water, unit: mg/L): {solubility}, Number of H-bond Acceptors: {hba}, Number of H-bond Donors: {hbd} and LogP: {logp}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the Molecular Weight (unit: g/mol): {molecular_weight}, Solubility (in water, unit: mg/L): {solubility}, Number of H-bond Acceptors: {hba}, Number of H-bond Donors: {hbd} and LogP: {logp}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Stability"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            energy_above_hull = data[topic][i]["label"]["Energy Above Hull (unit: eV/atom)"]
            if choices == "":
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Energy Above Hull (unit: eV/atom): {energy_above_hull}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Energy Above Hull (unit: eV/atom): {energy_above_hull}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Vector"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            lattice_angle = [data[topic][i]["label"]["Lattice Angle α (among 3 angles as [α, β, γ])"], data[topic][i]["label"]["Lattice Angle β (among 3 angles as [α, β, γ])"], data[topic][i]["label"]["Lattice Angle γ (among 3 angles as [α, β, γ])"]]
            if choices == "":
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Angle [α, β, γ]: {lattice_angle}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Angle [α, β, γ]: {lattice_angle}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Metal"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            band_gap = data[topic][i]["label"]["Band Gap (unit: eV)"]
            if choices == "":
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Band Gap (unit: eV): {band_gap}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Band Gap (unit: eV): {band_gap}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Ordering"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            total_magnetization = data[topic][i]["label"]["Total Magnetization (unit: µB/f.u.)"]
            if choices == "":
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Total Magnetization (unit: µB/f.u.): {total_magnetization}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Total Magnetization (unit: µB/f.u.): {total_magnetization}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["crystal_material"]["Density"]:
            mp_id = data[topic][i]["input"]["MP-id"]
            formula = data[topic][i]["input"]["Formula"]
            lattice_vector = [data[topic][i]["label"]["a in Lattice Vector [a, b, c] (unit: Å)"], data[topic][i]["label"]["b in Lattice Vector [a, b, c] (unit: Å)"], data[topic][i]["label"]["c in Lattice Vector [a, b, c] (unit: Å)"]]
            if choices == "":
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Vector [a, b, c] (unit: Å): {lattice_vector}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the MP-id: {mp_id}, Formula: {formula} and Lattice Vector [a, b, c] (unit: Å): {lattice_vector}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        elif [topic, label_name] in quest_lists["enzyme"]["Comprehensive"]:
            enzyme = data[topic][i]["input"]["Enzyme"]
            substrate = data[topic][i]["label"]["Substrate"]
            product = data[topic][i]["label"]["Product"]
            if choices == "":
                user_msg += f"Now knowing the Enzyme: {enzyme}, Substrate: {substrate} and Product: {product}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Now knowing the Enzyme: {enzyme}, Substrate: {substrate} and Product: {product}, think step by step and answer Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Choose one from the following choices and return A/B/C/D/E: {choices}\n LLM: "
        else:
            if choices == "":
                user_msg += f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM: "
            else:
                user_msg += f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}? Consider the following: {choices}\n LLM: "
        return user_msg


    def predict(self, model, data, examples, ans_name, GPT=True):
        if GPT:
            model_name = model[0]["model"]
            temp = model[0]["temperature"]
            openai.api_key = self.api_key
            openai.api_base = self.api_base

            # Decompose the dataset into question components and query one by one
            ans = self.initialize(data)

            checkpoint = False   # Skip answers that have already been generated. True: old answer exists; False: no old answer.
            for key in data.keys():
                topic = key
                for i in range(len(data[topic])):
                    entity = data[topic][i]
                    input = entity["input"]
                    input_name = list(input.keys())
                    input_value = list(input.values())
                    labels = entity["label"]
                    for name, value in zip(input_name, input_value):
                        ans[topic][i]["input"][name] = value
                    for label_name in labels.keys():
                        num = len(examples[topic])
                        example = []
                        for k in range(num):
                            example_ensemble = examples[topic][k]
                            example.append({
                                "example_input": [example_ensemble["input"][name] for name in input_name],
                                "example_label": example_ensemble["label"][label_name]
                            })

                        if os.path.exists(ans_name):
                            with open(ans_name, "r") as ans_file:
                                old_ans = json.load(ans_file)
                            if old_ans[topic][i]["label"][label_name]:
                                checkpoint = True
                            else:
                                checkpoint = False
                        else:
                            checkpoint = False

                        if not checkpoint:
                            ans[topic][i]["label"][label_name] = self.perform_task(ans, data, topic, i, input_name, input_value, label_name, example, model_name, temp, GPT=True)

                            with open(ans_name, "w") as json_file:
                                json.dump(ans, json_file, indent=4)
                        else:
                            ans = old_ans

            return ans
        else:   # LLaMA inference will be supported later
            raise NotImplementedError