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

# Running Inference

## Enzymes
Run the enzyme prediction experiments with one of the prompts
```shell
    python -m inference \
      --input "enzyme" \
      --model-config "GPT-3.5" \
      --agent-config "few-shot-CoT" \
      --n-examples 1
```

Change --model-config to GPT-4 or LLaMA-7B... to try other LLM models. Change --agent-config to few-shot-cot or few-shot-cot-critique... to try other prompts. Change --n-examples to other integer to set multi-shot.

## Small Molecules
Run the small molecule prediction experiments with one of the prompts
```shell
    python -m inference \
      --input "small_molecule" \
      --model-config "GPT-3.5" \
      --agent-config "few-shot-CoT" \
      --n-examples 1
```

Change --model-config to GPT-4 or LLaMA-7B... to try other LLM models. Change --agent-config to few-shot-cot or few-shot-cot-critique... to try other prompts. Change --n-examples to other integer to set multi-shot.

## Crystal Materials
Run the crystal material prediction experiments with one of the prompts
```shell
    python -m inference \
      --input "crtstal_material" \
      --model-config "GPT-3.5" \
      --agent-config "few-shot-CoT" \
      --n-examples 1
```

Change --model-config to GPT-4 or LLaMA-7B... to try other LLM models. Change --agent-config to few-shot-cot or few-shot-cot-critique... to try other prompts. Change --n-examples to other integer to set multi-shot.

## All
Run all prediction experiments with one of the prompts
```shell
    python -m inference \
      --input "All" \
      --model-config "GPT-3.5" \
      --agent-config "few-shot-CoT" \
      --n-examples 1
```

Change --model-config to GPT-4 or LLaMA-7B... to try other LLM models. Change --agent-config to few-shot-cot or few-shot-cot-critique... to try other prompts. Change --n-examples to other integer to set multi-shot.