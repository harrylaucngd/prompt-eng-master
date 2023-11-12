import json

# 读取JSON文件
for prompt in ["zero-shot", "expert", "few-shot", "zero-shot-CoT", "few-shot-CoT", "few-shot-CoT-critique"]:
    for i in range(3):
        name = f"results/1106/predict_dataset_{i+1}_small_molecule_{prompt}_gpt-3.5-turbo.json"
        with open(name, 'r') as file:
            data = json.load(file)

        # 遍历small_molecule中的每个元素
        for item in data['small_molecule']:
            # 如果"Drugability (Yes or No)"键存在，将其修改为"Drugability (0 to 10)": null
            if 'Drugability (0 to 10)' in item['label']:
                item['label']['Drugability (describe the degree of the drugability of this molecule using number from 0 to 10)'] = None
                del item['label']['Drugability (0 to 10)']

        # 写回JSON文件
        with open(name, 'w') as file:
            json.dump(data, file, indent=4)
