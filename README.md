# Prompt-Eng-Master
Repository for Prompt Engineering


# Setup

1. Install the required dependencies
    ```shell
    conda create -n decomp python=3.8
    pip install -r requirements.txt
    ```
2. Set the necessary env variables
    ```shell
    export PYTHONPATH=src/
    export OPENAI_API_KEY=<YOUR_API_KEY>
    export OPENAI_API_BASE=<YOUR_API_BASE>
    ```
    Or you can set them in config/api.json file:
    ```python
    {
        "openai_api": {
            "api_key": <YOUR_API_KEY>,
            "api_base": <YOUR_API_BASE>
        }
    }
    ```


# Running Inference

## Enzymes
Run the enzyme prediction experiments with one of the prompts
```shell
python -m run
```

Change the parameters below to try different datasets, models, prompts and other settings:
```python
setting_parameters = {
    # Input evaluation sets. Options: "enzyme", "small_molecule", "crystal_material" and "All".
    "input": "small_molecule",
    # Output direction. Append answer json file to results folder named after testing date.
    "output": "results/1014/",
    # LLM models. Options: "GPT-3.5", "GPT-4", "LLaMA2-7B", "LLaMA2-13B", "LLaMA2-70B".
    "model_config": "GPT-3.5",
    # Prompt configurations. Options: "zero-shot", "expert", "few-shot", "zero-shot-CoT", "few-shot-CoT", "few-shot-CoT-critique", "decomposed".
    "agent_config": "few-shot",
    # Number of few-shot examples (use multi-shot if set to >1). Default: 2.
    "n_examples": 2,
    # Number of llm-generated answers. Default: 2.
    "n_answers": 2
}
```