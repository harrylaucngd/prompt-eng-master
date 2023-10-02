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
    csv_file_path = "eval_dataset_small_molecule.csv"
    data_list = read_csv_to_list(csv_file_path)

data=[]

for i in range(np.size(data_list,0)-1):
    print("Molecule: "+str(i))
    elements=few_shot(data_list[i+1][0],data_list[i+1][1])
    Molecular_formula=elements[0]
    Molecular_weight=elements[1]
    H_bond_donors=elements[2]
    H_bond_acceptors=elements[3]
    Boiling_point=elements[4]
    Density=elements[5]
    logP=elements[6]
    tPSA=elements[7]
    Apolar_desolvation=elements[8]
    Polar_desolvation=elements[9]
    
    data.append([data_list[i+1][0],data_list[i+1][1],Molecular_formula,Molecular_weight,H_bond_donors,H_bond_acceptors,Boiling_point,Density,logP,tPSA,Apolar_desolvation,Polar_desolvation])

csv_file_path = "predict_dataset_small_molecule.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Molecule name','Smiles','Molecular formula','Molecular weight','H-bond acceptors','logP','tPSA','Apolar desolvation (kcal/mol)','Polar desolvation (kcal/mol)'])
    writer.writerows(data)

print("Data written to CSV successfully.")