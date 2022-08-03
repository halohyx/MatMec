import numpy as np
import pandas as pd
import glob
import os
import shutil
from matmec.core.latt import Latt

def extract_energy(OUTCAR):
    with open(OUTCAR, 'r') as f:
        for line in f.readlines():
            if 'free  energy   TOTEN' in line:
                # print(line)
                energy = float(line.split()[-2])
                # print(energy)
    return energy

# should be given!
work_root_folder = "basal"
# base_energy = -401.87691807


ev2mJ = 1.60217657E-16

work_path_list = glob.glob("%s/*/OUTCAR" % work_root_folder)
system_list = []
result_folder = os.path.abspath("%s/sfe_results" % work_root_folder)
os.makedirs(result_folder, exist_ok=True)
for work_path in work_path_list:
    system_list.append(os.path.join(result_folder,work_path.split("/")[-2]))
    shutil.copy(work_path, system_list[-1])

# get the area of x-y plane
poscar = Latt()
poscar.read_from_poscar("%s/POSCAR" % os.path.split(work_path_list[0])[0])
area = np.linalg.norm(np.cross(poscar.cell.lattvec[0], poscar.cell.lattvec[1]))*1E-20

# set the base energy of the gsfe curve, if it's empty, then it's set to the first of the calculation list
if base_energy is None:
    base_energy = extract_energy(system_list[0])

energy_dict = {}
system_list.sort()

for system in system_list:    
    energy_dict[int(system[-2:])/int(system_list[-1][-2:])] = (extract_energy(system) - base_energy)/area*ev2mJ
        
with open(os.path.join(result_folder, "sfe.out"), 'w') as f:
    for key in energy_dict:
        f.write("%s\t%s\n" % (key, energy_dict[key]))