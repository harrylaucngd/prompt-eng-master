import json

input = "small_molecule"
prompts = ["zero-shot", "expert", "zero-shot-CoT", "few-shot", "few-shot-CoT"]
for prompt in prompts:
    cap_loc = f"results/{input}/capability_{input}_{prompt}_gpt-3.5-turbo.json"
    F1_loc = f"results/{input}/F1_score_{input}_{prompt}_gpt-3.5-turbo.json"
    with open(cap_loc, 'r') as file:
        cap = json.load(file)
    with open(F1_loc, 'r') as file:
        f1 = json.load(file)
    
    for property in f1[input].keys():
        if f1[input][property] != 0:
            f1[input][property] = round(f1[input][property]*cap[input][property], 3)

    with open(F1_loc, "w") as file:
        json.dump(f1, file, indent=4)