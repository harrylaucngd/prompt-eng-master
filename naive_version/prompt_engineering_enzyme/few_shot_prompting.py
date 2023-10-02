#Few-shot Prompting
import os
import time
import openai

openai.api_key = "31D_RyHRqzqd"
openai.api_base = "https://api.openai-go.com/v1"

def few_shot(enzyme,entry_name,AA_sequence):
    messages = []
    properties = ['active site', 'Km (Mechaelis constant)', 'substrate', 'product']
    example_predict = ['C-114,H-222,E-224','2.93 mM','gamma-glutamyl-gamma-aminobutyrate','gamma-aminobutyrate']

    system_msg = "You are a highly acclaimed molecular and cellular biologist, renowned for your exceptional skills in deciphering amino acid sequences of enzymes and utilizing comprehensive databases to offer expert insights on the fundamental properties of enzymes. Your reputation precedes you, and your contributions to the field have been groundbreaking. As a leading expert, you possess an unrivaled understanding of enzyme functionalities, catalytic mechanisms, and their intricate roles in biological processes. Now, as you engage with various queries, your profound knowledge and expertise will undoubtedly pave the way for new discoveries and advancements in the realm of biochemistry and molecular biology."
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

    predict = []

    for i in range(4):
        time.sleep(15)
        print("round: "+str(i))
        user_msg = "Given the enzyme Gamma-glutamyl-gamma-aminobutyrate hydrolase PuuD (with entry name: PUUD_ECOLI) and its AA sequence: MENIMNNPVIGVVMCRNRLKGHATQTLQEKYLNAIIHAGGLPIALPHALAEPSLLEQLLPKLDGIYLPGSPSNVQPHLYGENGDEPDADPGRDLLSMAIINAALERRIPIFAICRGLQELVVATGGSLHRKLCEQPELLEHREDPELPVEQQYAPSHEVQVEEGGLLSALLPECSNFWVNSLHGQGAKVVSPRLRVEARSPDGLVEAVSVINHPFALGVQWHPEWNSSEYALSRILFEGFITACQHHIAEKQRL, what is the "+properties[i]+" of this enzyme? AI: "+example_predict[i]+". Given the enzyme: "+enzyme+" (with entry name: "+entry_name+") and its AA sequence: "+AA_sequence+", what is the "+properties[i]+" of this enzyme? AI: "
        
        messages.append({"role": "user", "content": user_msg})

        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        predict.append(chat_completion.choices[0].message.content)

    return predict