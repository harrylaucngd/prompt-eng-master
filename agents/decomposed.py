# decomposed prompting (based on ChemCrow)

import openai
import json
from chemcrow.agents import chemcrow
from agents.base import BaseModel


class DecomposedModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

    def decomposed(self, topic, input_name, input_value, label_name, model_name, temp, GPT=True):
        if GPT:
            # Define the user message
            user_msg = [
                {"role": "user", "content": f"Question: For {topic}, given the {input_name}: {input_value}, what is the {label_name}?\n LLM:"}
            ]
            chem_model = chemcrow.ChemCrow(
                model=model_name,
                temp=temp,
                max_iterations=2,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                api_keys={}
            )
            response = chem_model.run(user_msg)
            return response
        else:   # LLaMA inference will be supported later
            return "N/A"

    def perform_task(self, topic, input_name, input_value, label_name, example, model_name, temp, GPT=True):
        return self.decomposed(topic, input_name, input_value, label_name, model_name, temp, GPT=True)