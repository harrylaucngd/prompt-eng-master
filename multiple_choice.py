import json
from inference import data_loader


def multiple_choice(label_name, ans):
    # What's the reasonable algorithm to define choices?
    rules = choice_range[label_name]
    unit = rules[2]
    if rules[0] == "0":  # Fixed
        choice_list = rules[1]
        choices = ["A: %s %s, B: %s %s, C: %s %s, D: %s %s, E: %s %s" % (choice_list[0], unit, choice_list[1], unit, choice_list[2], unit, choice_list[3], unit, choice_list[4], unit)]
    elif rules[0] == "1":   # Variable value
        choice_list = []
        gap = rules[1]
        choices = ["A: %s, B: %s, C: %s, D: %s, E: %s" % (unit, unit, unit, unit, unit)]
    else:   # Variable invertal
        choice_list = []
        choices = ["A: %s, B: %s, C: %s, D: %s, E: %s" % (unit, unit, unit, unit, unit)]

    return choices


def answer(ans, choices):
    return ans


choice_range = {
    "Molecular Weight (unit: g/mol)": [2, 10, "(unit: g/mol)"],
    "Number of H-bond Acceptors": [1, 2, ""],
    "Number of H-bond Donors": [1, 2, ""],
    "a in Lattice Vector [a, b, c] (unit: Å)": [2, 1, "(unit: Å)"],
    "b in Lattice Vector [a, b, c] (unit: Å)": [2, 1, "(unit: Å)"],
    "c in Lattice Vector [a, b, c] (unit: Å)": [2, 1, "(unit: Å)"],
    "Lattice Angle α (among 3 angles as [α, β, γ])": [0, ["α<90", "90<=α<100", "100<=α<110", "110<=α<120", "α>120"], ""],
    "Lattice Angle β (among 3 angles as [α, β, γ])": [0, ["β<90", "90<=β<100", "100<=β<110", "110<=β<120", "β>120"], ""],
    "Lattice Angle γ (among 3 angles as [α, β, γ])": [0, ["γ<90", "90<=γ<100", "100<=γ<110", "110<=γ<120", "γ>120"], ""],
    "Space Group Number": [1, 15, ""],
    "Number of Amino Acids": [2, 50, ""],
    "Melting Point (unit: ℃)": [2, 20, "(unit: ℃)"],
    "Density (unit: g/cm3)": [2, 0.1, "(unit: g/cm3)"],
    "Solubility (in water, unit: mg/L)": [0, ["<1", "1~10", "10~100", "100~1000", ">1000"], "(in water, unit: mg/L)"],
    "LogP": [2, 0.5, ""],
    "Crystal Density (unit: g/cm3)": [2, 0.5, "(unit: g/cm3)"],
    "Formation Energy (unit: eV/atom)": [2, 0.5, "(unit: eV/atom)"],
    "Energy Above Hull (unit: eV/atom)": [2, 0.05, "(unit: eV/atom)"],
    "Band Gap (unit: eV)": [2, 0.5, "(unit: eV)"],
    "Total Magnetization (unit: µB/f.u.)": [2, 1, "(unit: µB/f.u.)"]
}


if __name__ == "__main__":
    datasets = ["data/eval_dataset_small_molecule.csv", "data/eval_dataset_enzyme.csv", "data/eval_dataset_crystal_material.csv"]
    data_dict = data_loader(datasets)

    for topic in data_dict.keys():
        for i in range(len(data_dict[topic])):
            entity = data_dict[topic][i]
            labels = entity["label"]
            labels = {key: value for key, value in labels.items() if key in choice_range.keys()}
            data_dict[topic][i]["label"] = labels


    with open("data/multiple_choice.json", "w") as file:
        json.dump(data_dict, file, indent=4)
'''
    cot_classification_name = "data/cot_classification.json"
    with open(cot_classification_name, 'r') as file:
        quest_lists = json.load(file)
    quest_list = quest_lists["All"]

    for topic in data_dict.keys():
        for i in range(len(data_dict[topic])):
            entity = data_dict[topic][i]
            input = entity["input"]
            input_name = list(input.keys())
            input_value = list(input.values())
            labels = entity["label"]
            for label_name in labels.keys():
                if (([topic, label_name] in quest_list["Numerical & Logical"]) or ([topic, label_name] in quest_list["Numerical & Experimental"])):
                    choices = multiple_choice(label_name, data_dict[topic][i]["label"][label_name])
                    data_dict[topic][i]["label"][label_name] = choices
                    data_dict[topic][i]["label"][label_name].append(answer(data_dict[topic][i]["label"][label_name], choices))
                #else:
                    #del data_dict[topic][i]["label"][label_name]

    with open("data/multiple_choice.json", "w") as file:
        json.dump(data_dict, file, indent=4)
        '''