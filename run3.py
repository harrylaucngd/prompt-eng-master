from inference import infer


setting_parameters = {
    # Input evaluation sets. Options: "enzyme", "small_molecule", "crystal_material" and "All".
    "input": "small_molecule",
    # Output direction. Append answer json file to results folder named after testing date.
    "output": "results/small_molecule/",
    # LLM models. Options: "GPT-3.5", "GPT-4", "LLaMA2-7B", "LLaMA2-13B", "LLaMA2-70B".
    "model_config": "GPT-3.5",
    # Prompt configurations. Options: "zero-shot", "expert", "few-shot", "zero-shot-CoT", "few-shot-CoT", "few-shot-CoT-critique", "decomposed".
    "agent_config": "zero-shot-CoT",
    # Number of few-shot examples (use multi-shot if set to >1). Default: 2.
    "n_examples": 2,
    # Number of llm-generated answers. Default: 2.
    "n_answers": 3
}

if __name__ == "__main__":

    parameters = list(setting_parameters.values())
    infer(*parameters)