#Few-shot Prompting
import os
import time
import openai

openai.api_key = "31D_RyHRqzqd"
openai.api_base = "https://api.openai-go.com/v1"

def few_shot(Molecule_name,Smiles):
    messages = []
    properties = ['molecular formula','molecular weight','number of H-bond donors','number of H-bond acceptors','Boiling Point','Density','logP (octanol-water partition coefficient)','tPSA (Topological polar surface area)','Apolar desolvation (kcal/mol)','Polar desolvation (kcal/mol)']
    example_predict = ['C3H8O','60.1 g/mol','1','1','82.3 °C','0.785 g/cm3','0.387','20 Å²','-0.41','-2.38']

    system_msg = "You are a highly regarded and accomplished chemist, celebrated for your exceptional proficiency in deciphering SMILES sequences of various substances and leveraging comprehensive databases like PubChem and Zinc to offer expert insights on the fundamental properties of these compounds. Your reputation as a leading expert in the field of chemistry precedes you, and your contributions to the domain have been revolutionary. As you delve into diverse inquiries, your profound knowledge and acumen in substance analysis will undoubtedly fuel novel discoveries and advancements in the realms of pharmaceuticals, materials science, and chemical research, making an indelible impact on the world of chemistry."
    user_msg = "Who are you?"

    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_msg,
            "role": "user", "content": user_msg}]
    )
    answer = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    messages.append({})

    predict = []

    for i in range(10):
        time.sleep(10)
        print("round: "+str(i))
        user_msg = "Given the molecule Isopropanol and its Smiles: CC(C)O, what is the "+properties[i]+" of this molecule? AI: "+example_predict[i]+". Given the molecule: "+Molecule_name+" and its Smiles: "+Smiles+", what is the "+properties[i]+" of this molecule? AI: " #对齐版在后面添加了说明：“Try your best to directly give an effective answer. If not able, just return N/A and nothing else.”
        
        messages[3]={"role": "user", "content": user_msg}

        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        predict.append(chat_completion.choices[0].message.content)

    return predict