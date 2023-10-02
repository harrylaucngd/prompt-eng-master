#Expert Prompting
import os
import openai

openai.api_key = "31D_RyHRqzqd"
openai.api_base = "https://api.openai-go.com/v1"

def expert(enzyme,entry_name,AA_sequence):

    # Define the system message
    system_msg = "You are these three expert: 1. Dr. Sarah Johnson - Biochemist specializing in enzyme kinetics and substrate specificity. 2. Dr. Michael Patel - Enzymologist with expertise in enzyme-substrate interactions and enzymatic reactions. 3. Dr. Emily Chen - Molecular biologist with a focus on enzyme function and protein engineering. Assemble their opinions."

    # Define the user message
    user_msg = "Solve this problem: Given the enzyme KPSF_ECOL6 (with entry on Uniprot: Q8FDQ2), Try to analyse the function of this enzyme and tell me what is the substrate and product of this enzyme? You can search for information on Uniprot."
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{ "role": "system", "content": "",
            "role": "user", "content": user_msg }]
    )
    return chat_completion.choices[0].message.content
