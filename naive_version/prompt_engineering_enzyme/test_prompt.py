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
    csv_file_path = "eval_dataset_enzyme.csv"
    data_list = read_csv_to_list(csv_file_path)

data=[]

for i in range(np.size(data_list,0)-1):
    print("Enzyme: "+str(i))
    elements=few_shot(data_list[i+1][0],data_list[i+1][1],data_list[i+1][2])
    active_site=elements[0]
    Km=elements[1]
    substrate=elements[2]
    product=elements[3]
    
    data.append([data_list[i+1][0],data_list[i+1][1],data_list[i+1][2],active_site, Km, substrate, product])

csv_file_path = "predict_dataset_enzyme.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['enzyme','entry_name','AA_sequence','Active Site', 'Km', 'Substrate', 'Product'])
    writer.writerows(data)

print("Data written to CSV successfully.")