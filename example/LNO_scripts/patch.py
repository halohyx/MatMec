import os
import numpy as np
from monty.serialization import loadfn
import shutil
from pymatgen.io.vasp.inputs import Incar

calc_lists = [1]

calc0 = "0-Li_GB"
calc1 = "1-Li_Bulk"
calc2 = "2-Li_Surf"
calc3 = "3-TM_GB"
calc4 = "4-TM_Bulk"
calc5 = "5-TM_Surf"
calc6 = "6-Li_Tensile"
calc7 = "7-TM_Tensile"
calc8 = "8-Li_RGS"
calc9 = "9-TM_RGS"

calcs = [calc0, calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9]

def patchMag(calc, magmom):
    magmom = np.array(magmom)
    if calc == "0-Li_GB":
        magmom[:8] = 0.0
        # magmom[8:12] = magmom[8:12]
        magmom[8:12] = 0.0
        magmom[12:24] = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 0.8, 1.8, 1.8, 1.8, 1.8]
        magmom[24:] = 0.0
    elif calc == "1-Li_Bulk":
        magmom[:10] = 0.0
        # magmom[10:12] = magmom[10:12]
        magmom[10:12] = 0.0
        magmom[12:24] = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
        magmom[24:] = 0.0
    elif calc == "2-Li_Surf":
        magmom[:4] = 0.0
        # magmom[4:6] = magmom[4:6]
        magmom[4:6] = 0.0
        magmom[6:12] = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
        magmom[12:] = 0.0
    elif calc == "3-TM_GB":
        magmom[:12] = 0.0
        # magmom[12:16] = magmom[12:16]
        magmom[12:16] = 0.0
        magmom[16:24] = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
        magmom[24:] = 0.0
    elif calc == "4-TM_Bulk":
        magmom[:12] = 0.0
        # magmom[12:14] = magmom[12:14]
        magmom[12:14] = 0.0
        magmom[14:24] = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
        magmom[24:] = 0.0
    elif calc == "5-TM_Surf":
        magmom[:6] = 0.0
        # magmom[6:8] = magmom[6:8]
        magmom[6:8] = 0.0
        magmom[8:12] = [1.8, 1.8, 1.8, 1.8]
        magmom[12:] = 0.0
    return list(magmom)

def dict_del(dictionary, *args):
    for key in args:
        if key in dictionary:
            dictionary.pop(key)

root_dir = os.getcwd()
for calc in calc_lists:
    os.chdir(calcs[calc])
    incar = Incar.from_file("INCAR")
    # if incar["ENCUT"] == 600:
    incar.update({"ENCUT": 600,
                    "ALGO": "NORMAL",
                    "PREC": "Normal",
                    "MAGMOM": patchMag(calcs[calc], incar["MAGMOM"]),
                    "LORBIT": 11,
                    "IBRION": 2,
                    "NELMIN" : 6})
                    # "AMIX": 0.10,
                    # "BMIX": 0.0001,
                    # "BMIX_MAG": 0.0001})
    dict_del(incar, "AMIX", "BMIX", "BMIX_MAG", "EDIFFG")
    # dict_del(incar, "EDIFFG")
    incar.write_file("INCAR")
    # print(incar)
    # shutil.copy("CONTCAR", "POSCAR")
    os.system("sbatch ~/bin/vasp_630.slurm")
    del incar
    os.chdir(root_dir)
