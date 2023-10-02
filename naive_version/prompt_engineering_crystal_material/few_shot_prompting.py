#Few-shot Prompting
import os
import time
import openai

openai.api_key = "31D_7u8GomPK"
openai.api_base = "https://api.openai-go.com/v1"

def few_shot(Crystal_material):
    messages = []
    properties = ['Ground State Phase','Competing Phases','ΔfH','Decomposition Energy']
    example_predict = ['Al2O3','Al + Al5O8','-3.473 eV/atom','0.146 eV/atom']

    system_msg = "You are a highly regarded and accomplished chemist, celebrated for your exceptional proficiency in the intricate analysis of the composition of crystal materials. Your reputation as a leading expert in the field of crystallography is widely acknowledged, and your contributions have pushed the boundaries of our understanding of these complex structures. With your extensive knowledge, you are well-equipped to provide detailed insights into the fundamental properties of crystal materials. Your expertise is invaluable in advancing materials science, semiconductor research, and countless other domains that rely on the unique characteristics of crystal structures."
    user_msg = "Who are you?"

    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    chat_completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_msg,
            "role": "user", "content": user_msg}]
    )
    answer = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    messages.append({})

    predict = []

    for i in range(4):
        time.sleep(10)
        print("round: "+str(i))
        user_msg = "Given the crystal material Al2O3, what is the "+properties[i]+" of this crystal material? AI: "+example_predict[i]+". Given the crystal material: "+Crystal_material+", what is the "+properties[i]+" of this crystal material? AI: "

        messages[3]={"role": "user", "content": user_msg}

        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        predict.append(chat_completion.choices[0].message.content)

    return predict