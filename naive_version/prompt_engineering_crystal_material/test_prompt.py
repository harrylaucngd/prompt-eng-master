import argparse
import csv
import os
import numpy as np
from expert_prompting import expert
from few_shot_prompting import few_shot

def read_csv_to_list(file_path):
    data_list = []
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data_list.append(row)
    return data_list

if __name__ == "__main__":
    csv_file_path = "eval_dataset_crystal_material.csv"
    data_list = read_csv_to_list(csv_file_path)

data=[]

for i in range(np.size(data_list,0)-1):
    print("Crystal Material: "+str(i))
    elements=few_shot(data_list[i+1][0])
    Ground_state_phase=elements[0]
    Competing_phases=elements[1]
    ΔfH=elements[2]
    Decomposition_energy=elements[3]
    
    data.append([data_list[i+1][0],Ground_state_phase,Competing_phases,ΔfH,Decomposition_energy])

csv_file_path = "predict_dataset_crystal_material.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Composition','Ground State Phase','Competing Phases','ΔfH','Decomposition Energy'])
    writer.writerows(data)

print("Data written to CSV successfully.")