# Script for extract results from a set of DFT calculations based on doping different element. 
# The calculations are based on the LiNiO2 (012)<12-1> CSL twin grain boundary structure.
# Arthor: Yixuan Hu
# Email: yixuanhu97@sjtu.edu.cn
# Date: 2023-04-02
# 
# There are two sites to dope the elements, one is on the grain boundary (GB), the other is on the neighboring sites of the GB.
# The calculations about doping at the GB is under 2-012_12-1_SU_GB/1-D3_IVDW11 folder, and the calculations about doping at the neighboring sites of the GB is under 2-012_12-1_SU_GB_Neighbor/1-D3_IVDW11 folder.
# And for each doping site, there are two types of doping, one is to replace the Li atom, the other is to replace the transition metal (Ni) atom.
# 
# --------------------------------------------------------------------------------------------------------------------- #
# There are 12 basic calculations in total for each element doping, and the calculations are under the folder of the element name.
# The 12 basic calculation name is defined as follows:
# 0-Li_GB, 3-TM_GB under 2-012_12-1_SU_GB folder indicate the doping at GB to replace the Li and TM atom, respectively.
# 1-Li_Bulk, 4-TM_Bulk under 2-012_12-1_SU_GB folder indicate the doping in Bulk structure to replace the Li and TM atom, respectively.
# 2-Li_Surf, 5-TM_Surf under 2-012_12-1_SU_GB folder indicate the doping of the surface slab to replace the Li and TM atom, respectively.
# 0-Li_GB, 3-TM_GB under 2-012_12-1_SU_GB_Neighbor folder indicate the doping at the neighbor sites of the GB to replace the Li and TM atom, respectively.
# 1-Li_Bulk, 4-TM_Bulk under 2-012_12-1_SU_GB_Neighbor folder indicate the doping in bulk site of the GB structure (notice, it's not the real bulk structure, but one site far from the GB in the GB structure to simulate a bulk doping) to replace the Li and TM atom, respectively.
# 2-Li_Surf, 5-TM_Surf under 2-012_12-1_SU_GB_Neighbor folder indicate the doping of the surface slab of the neighbor sites of GB to replace the Li and TM atom, respectively.
#
# 12 related energy variables of these 12 basic calculations are defined as follows:
# ener_GB_Li, ener_GB_TM, ener_realBulk_Li, ener_realBulk_TM, ener_Surf_Li, ener_Surf_TM, 
# ener_GB_neigh_Li, ener_GB_neigh_TM, ener_gbBulk_Li, ener_gbBulk_TM, ener_Surf_neigh_Li, ener_Surf_neigh_TM
# 
# There are 5 properties that can be derived from the energies of 12 basic calculations:
# 1. grain boundary formation energy (GBFE), which is defined as the difference between the energy of the GB and the energy of the GB bulk structure:
#    GBFE[J/m2] = 1/2*(ener_GB/area_GB - ener_gbBulk/area_gbBulk)
#    *** area_GB and area_gbBulk are the area of the GB and the GB bulk structure, respectively.
# 2. solute segregation energy (SSE) at GB, which is defined as the difference between the energy of the GB and the energy of the bulk structure:
#    SSE[eV/atom]  = 1/2*( (ener_GB + 2*ener_realBulk_ref) - (ener_GB_ref + 2*ener_realBulk) )
#    *** ener_GB_ref and ener_realBulk_ref are the energy of the undoped GB and the bulk structure, respectively.
#    *** There are two doped atoms in the GB structure, but only one doped atom in the bulk structure, so the energy of the bulk structure is divided by 2.
# 3. SSE at neighbor doping sites, which is defined as the difference between the energy of the neighbor doped GB and the energy of the bulk structure:
#    SSE_neigh[eV/atom]  = 1/4*( (ener_GB_neigh + 4*ener_realBulk_ref) - (ener_GB_ref + 4*ener_realBulk) )
#    *** There are four doped atoms in the neighbor doping sites structure, but only one doped atom in the bulk structure, so the energy of the bulk structure is divided by 4.
# 4. seperation work (SW), which is defined as the difference between the energy of the GB and the energy of the surface slab:
#    SW[J/m2] = 1/2*(ener_GB/area_GB - 2*ener_Surf/area_Surf)
#    *** One GB structure splitted into two free surfaces.
#    *** area_GB and area_Surf are the area of the GB and the surface slab, respectively.
# 5. SW at neighbor doping sites, which is defined as the difference between the energy of the neighbor doped GB and the energy of the surface slab:
#    SW_neigh[J/m2] = 1/2*(ener_GB_neigh/area_GB_neigh - 2*ener_Surf_neigh/area_Surf_neigh)
# 
# --------------------------------------------------------------------------------------------------------------------- #
# 2 type of tensile calculation (Normal and RGS ) based on different doping sites is defined as follows:
# 6-Li_Tensile, 7-TM_Tensile under 2-012_12-1_SU_GB folder indicate the tensile strain calculation to replace the Li and TM atom, respectively.
# 8-Li_RGS, 9-TM_RGS under 2-012_12-1_SU_GB folder indicate the RGS calculation to replace the Li and TM atom, respectively.
#
# Normal tensile:
# Normal tensile is done by enlongating the GB along the GB normal direction (Z in this case)
# 1. extract the energy from diffrent strain calculation to get the strain-energy curve
# 2. calculate the differential of the strain-energy curve to get the stress-strain curve
# *** The defferential will be calculated using central difference method.
# *** Another way will be considered as using cubic spline to fit the strain-energy curve and then calculate the differential.
# RGS tensile:
# RGS tensile is done by manually make displacement between two grains (same as the surfaces we used) in the GB structure.
# 1. extract the energy from diffrent displacement calculation to get the displacement-energy curve
# 2. plot the discret displacement-energy dataset and using the universal binding energy relationship (UBER) to fit it
# *** The universal binding energy relationship (UBER) is defined as:
#     E(delta) = g(a)*|Eeb|
#     where g(a) = -(1+a)*exp(-a)
#     and Eeb is the binding energy of the grain boundary (defined as where the total energy doesn't change with the displacement)
#     and in g(a), a = delta/l, which is the reduced displacement
#     and l is the characteristic length, which is the curvature when displacement is zero
#     and delta is the displacement
# --------------------------------------------------------------------------------------------------------------------- #
# The results will be stored in json files
# eleSum.json for the summary of each element and will be stored in element folder
# allSum.json for the results of each calculation and will be stored in root folder
# You will use element_lists and calc_lists to define the elements and calculations you want to extract the results and update the results json.

# import modules
import os
import shutil
import pandas as pd
import numpy as np
from matmec.core import Latt
from monty.serialization import loadfn
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import json
from glob import glob
import warnings
from copy import deepcopy

# define elements and calculations that you want to extract and update the results json
element_lists = ["Ta"]
calc_lists = [1] #,1,2,3,4,5]
doping_sites = ["neigh"]

# the root path of each doping calculation
doping_sites_dirs_dict = {
    "GB": "2-012_12-1_SU_GB/1-D3_IVDW11/",
    "neigh": "3-012_12-1_SU_GB_neighbor/1-D3_IVDW11/"
}

paraDict = loadfn("./parameters.json")

# the name of the calculations folders
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

# dictionary of output property name of the calculations
calc_dict = { 
"GB": {calc0: "GB_Li", 
        calc1: "realBulk_Li", 
        calc2: "Surf_Li", 
        calc3: "GB_TM", 
        calc4: "realBulk_TM", 
        calc5: "Surf_TM", 
        calc6: "GB_tensile_Li", 
        calc7: "GB_tensile_TM", 
        calc8: "GB_RGS_Li", 
        calc9: "GB_RGS_TM"},
"neigh": {calc0: "GB_neigh_Li", 
        calc1: "gbBulk_Li", 
        calc2: "Surf_neigh_Li", 
        calc3: "GB_neigh_TM", 
        calc4: "gbBulk_TM", 
        calc5: "Surf_neigh_TM", 
        calc6: "GB_tensile_neigh_Li", 
        calc7: "GB_tensile_neigh_TM", 
        calc8: "GB_RGS_neigh_Li", 
        calc9: "GB_RGS_neigh_TM"}
}


# a list object of all calculations
calcs = [calc0, calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9]

# define the path of the results json, if it doen't exist, create one for latter dump
if os.path.exists("./results.json"):
    resultsDict = loadfn("./results.json")
else:
    resultsDict = {}
    with open("./results.json", "w") as f:
        json.dump(resultsDict, f, indent=4)

# define the default empty results dictionary for each element
emptyEleRes = {
    "GB_Li": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "realBulk_Li": {"energy": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "Surf_Li": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "GB_TM": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "realBulk_TM": {"energy": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "Surf_TM": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "GB_neigh_Li": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "Surf_neigh_Li": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "GB_neigh_TM": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "Surf_neigh_TM": {"energy": 0.0, "area": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "gbBulk_Li": {"energy": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "gbBulk_TM": {"energy": 0.0, "energyPerArea": 0.0, "updateTime": "0000-00-00:00:00:00"},
    "outputProperties": {
        "updateTime": "0000-00-00:00:00:00",
        "GBFE_Li": "Not applicable",
        "SSE_Li": "Not applicable",
        "SW_Li": "Not applicable",
        "SSE_neigh_Li": "Not applicable", 
        "SW_neigh_Li": "Not applicable",
        "GBFE_TM": "Not applicable",
        "SSE_TM": "Not applicable",
        "SW_TM": "Not applicable",
        "SSE_neigh_TM": "Not applicable",
        "SW_neigh_TM": "Not applicable",
        }
}

# define the eV to J conversion factor
eV2J = 1.602176634E-19

# name of the summary json
summaryFile = "allSum.json"

# def the function to extract energy from OUTCAR
def extract_energy(OUTCAR):
    with open(OUTCAR, 'r') as f:
        for line in f.readlines():
            if 'free  energy   TOTEN' in line:
                # print(line)
                energy = float(line.split()[-2])
                # print(energy)
    return energy

# def the function to extract energy from OSZICAR
def extract_energy_osz(oszicar: str = "OSZICAR"):
    with open(oszicar) as f:
        line = f.readline()
        while line:
            if len(line.split("F=")) > 1:
                energy = float(line.strip().split()[2])
            line = f.readline()
    return energy

# def the function to extract the volume from OUTCAR
def extract_volume(OUTCAR):
    # a = poscar()
    # a.readFromPOSCAR(CONTCAR)
    # volume = np.dot(np.cross(a.latticeVectors[0], a.latticeVectors[1]), a.latticeVectors[2])
    # atoms = len(a.atomsList)
    with open(OUTCAR, 'r') as f:
        for line in f.readlines():
            if 'volume of cell :' in line:
                # print(line)
                volume = float(line.split()[-1])   

    return volume

# def the function to extract the stress from OUTCAR
def extract_stress(OUTCAR):
    with open(OUTCAR, 'r') as f:
        for line in f.readlines():
            if 'in kB' in line:
                # print(line)
                stress = np.array(line.split()[2:], dtype=float)  
    return stress

# def the function to extract all information from OUTCAR, including energy, volume and stress
def extract_all(OUTCAR):
    for line in f.readlines():
        if 'in kB' in line:
            # print(line)
            stress = np.array(line.split()[2:], dtype=float)  
        elif 'free  energy   TOTEN' in line:
            # print(line)
            energy = float(line.split()[-2])
        elif 'volume of cell :' in line:
            # print(line)
            volume = float(line.split()[-1]) 
    return energy, volume, np.array(stress, dtype=float)

# def the function to Get area from one POSCAR style structure file
def get_area(poscar):
    tmp = Latt.read_from_poscar(poscar)
    return np.linalg.norm(np.cross(tmp.cell.a, tmp.cell.b))

# def function to get forward difference
def forward_difference(x: list, y: list):
    '''
    To get the forward_difference of x and y
    n - 1 dimension, a None is filled in the ending of the result to get a n dimensional list
    '''
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        raise ValueError('x, y should be the same length list')
    else:
        forward_diff = []
        for i in range(len(y)-1):
            diff = (y[i+1] - y[i])/(x[i+1] - x[i])
            forward_diff.append(diff)
        forward_diff.append(None)
    return forward_diff

# def the function to get backward difference
def backward_difference(x: list, y: list):
    '''
    To get the backward_difference of x and y
    n - 1 dimension, a None is filled in the beginning of the result to get a n dimensional list
    '''
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        raise ValueError('x, y should be the same length list')
    else:
        backward_diff = [None]
        for i in range(len(y)-1):
            diff = (y[i+1] - y[i])/(x[i+1] - x[i])
            backward_diff.append(diff)
        
    return backward_diff

# def the function to get central difference
def central_diff(x: list, y: list):
    '''
    To get the backward_difference of x and y
    n - 2 dimension, the two ends are approximated by forward diff and backward diff
    '''
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        raise ValueError('x, y should be the same length list')
    else:
        forward_diff = forward_difference(x, y)
        backward_diff = backward_difference(x, y)
        central_diff = []
        for i in range( len(forward_diff) ):
            if forward_diff[i] == None:
                central_diff.append(backward_diff[i])
            elif backward_diff[i] == None:
                central_diff.append(forward_diff[i])
            else:
                central_diff.append((forward_diff[i] + backward_diff[i])/2)
        
        return central_diff

# def the function to get the cubic spline interpolate
def cubic_spline(x: list, y: list):
    '''
    Returns the cubic spline interpolate, and the corresponding difference at point xi
    '''
    from scipy.interpolate import CubicSpline

    x = np.array(x)
    y = np.array(y)
    cs = CubicSpline(x, y)

    return cs, cs(x, 1)

# def the function to get the curvature of three points
def Curvature(x,y):
    import numpy.linalg as LA
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """
    t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(LA.inv(M),x)
    b = np.matmul(LA.inv(M),y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

# given the RGS raw results, plot the figure 
def RGS_plotting(disp, energy):
    '''
    Give the RGS raw results, plot the figure
    '''
    # cubicspline
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(disp, energy)

    xdisp = np.linspace(-1, 6, 100)

    Eb0Index = np.where(disp==0)[0][0]
    Eb0 = energy[Eb0Index]
    CurvX = disp[np.array([Eb0Index-1, Eb0Index, Eb0Index+1])]
    CurvY = energy[np.array([Eb0Index-1, Eb0Index, Eb0Index+1])]
    # l = np.sqrt(np.abs(Eb0)/np.abs(Curvature(CurvX, CurvY)[0]))
    l = np.sqrt(np.abs(Eb0)/np.abs(cs(0, 2)))
    l = 0.78
    a = xdisp/l
    Eb = -(1+a)*np.exp(-a)*np.abs(Eb0)
    fig = plt.figure(figsize=(6, 6))
    style='bmh'
    mpl.rcParams['font.family'] = "Arial"
    plt.style.use(style)
    ax = fig.gca()
    ax.scatter(x=disp, y=energy, s = 100, c = "m", marker = "+", label="Raw Data")
    ax.plot(xdisp, Eb, label="UBER fitting", c="green", lw=3, ls="-", alpha=0.8)
    ax.plot(xdisp, cs(xdisp, 0), label="CubicSpline fitting", c="b", lw=2, ls="--") 
    ax.legend(fontsize=10, loc=0)

    title_font = {'family': 'calibri', 'size': 14}
    superTitle_font = {'family': 'arial', 'size': 14}
    ax.set_title(r"$Rigid\ Shift\ Grain\ Results$", fontdict=superTitle_font)
    ax.set_xlabel(r"$Displacement\ (\AA)$", fontdict=title_font)
    ax.set_ylabel(r"$Energy\ [J/m2]$", fontdict=title_font)
    plt.savefig("RGS.png")

# print time in a 2023-04-01-21:13:00 format
def current_time():
    import time
    return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

# compare two time string and tell which one is later
def compare_time(time1, time2):
    import time
    time1 = time.strptime(time1, "%Y-%m-%d-%H:%M:%S")
    time2 = time.strptime(time2, "%Y-%m-%d-%H:%M:%S")
    if time1 > time2:
        return 1
    elif time1 < time2:
        return -1
    else:
        return 0

# def a function to get the difference between two lists
def get_list_diff(a, b):
    return list(set(a)-set(b))

# extract the results of normal tensile
def extract_tensile():

    tensileResultsList = glob("./*/*/*/OUTCAR", recursive=True)
    strainlist = []
    energylist = []
    volumelist = []
    stresslist = []
    for ii in tensileResultsList:
        strain = ii.split('/')[-2].split("_")[-1]
        energy, volume, stress = extract_all(ii)
        strainlist.append(strain)
        energylist.append(energy)
        volumelist.append(volume)
        stresslist.append(stress)
    strainlist = np.array(strainlist, dtype=float)
    strainlist = strainlist - 1.0
    energylist = np.array(energylist, dtype=float)
    volumelist = np.array(volumelist, dtype=float)
    stresslist = np.array(stresslist, dtype=float)
    
    # sort
    sortingMask = strainlist.argsort()
    strainlist = strainlist[sortingMask]
    energylist = energylist[sortingMask]
    volumelist = volumelist[sortingMask]
    stresslist = stresslist[sortingMask]
    
    eV2J = 1.60218E-19
    
    # the energy-stress way, using central differential and cubic spine differential
    energyDiff = eV2J*1E21*(energylist - energylist[0])/volumelist
    stressCentralDiff = central_diff(strainlist, energyDiff)
    _, stressCubicSpineDiff = cubic_spline(strainlist, energyDiff)
    
    # generate result dictionary
    tmpDict = {}
    tmpDict["strain"] = list(strainlist)
    tmpDict["TotalEnergy[eV]"] = list(energylist)
    tmpDict["volume[A3]"] = list(volumelist)
    tmpDict["Energy[J/m3]"] = list(energyDiff)
    tmpDict["StressCentralDiff[GPa]"] = list(stressCentralDiff)
    tmpDict["StressCubicSpine[GPa]"] = list(stressCubicSpineDiff)
    
    stressVariants = ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'ZX']
    for ii in range(len(stressVariants)):
        tmpDict[stressVariants[ii]] = list(stresslist[:, ii]/10)
    
    df = pd.DataFrame(tmpDict)
    df.to_excel("results.xlsx")
    
    return tmpDict

# extract the results of RGS
def extract_RGS():
    RGSResultsList = glob("./*/*/*/OSZICAR", recursive=True)
    displist = []
    energylist = []
    area = get_area("POSCAR")
    print(area)
    for ii in RGSResultsList:
        disp = ii.split('/')[-2].split("_")[-1]
        # the energy is obtained from OSZICAR
        energy = extract_energy_osz(ii)
        displist.append(disp)
        energylist.append(energy)
        energyPerAlist = energylist/area

    # sort
    displist = np.array(displist, dtype=float)
    sortingMask = displist.argsort()
    displist = displist[sortingMask]
    energyPerAlist = np.array(energyPerAlist, dtype=float)[sortingMask]
    energylist = np.array(energylist, dtype=float)[sortingMask]

    eV2J = 1.60218E-19

    # use the energy of the final item as the base to generate the curve
    energyDiff = eV2J*1E20*(energyPerAlist - energyPerAlist[-1])
    
    # generate result dictionary
    tmpDict = {}
    tmpDict["disp"] = list(displist)
    tmpDict["Energy[J/m2]"] = list(energyDiff)

    df = pd.DataFrame(tmpDict)
    df.to_excel("results.xlsx")
    
    # plot the results
    RGS_plotting(np.array(df["disp"]), np.array(df["Energy[J/m2]"]))

    return tmpDict


# def a function, given the element, calc_lists, doping_sites, extract energy from OSZICAR, area from CONTCAR, calculate the energy per area
# 1. store the energy, area, energyPerArea in a dictionary named eleSumDict
# 2. save the eleSumDict as a json file named eleSum.json in each element directory
# Input: element, calc_lists, doping_sites, summaryJson (Use the emptyEleRes as the template)
# Return: eleSumDict
# get the complete eleSum.json of one element
def extract_all_info(element, calc_lists, doping_sites, overwrite = False):
    eleSummaryJson = "eleSum.json"

    eleSumDict = None
    rootPath = os.getcwd()
    
    # a tag help to initialize the eleSumDict
    helpTag = True

    # loop over doping sites
    for dopsite in doping_sites:
        dopeRootPath = os.path.abspath(doping_sites_dirs_dict[dopsite])
        whole_doping_list = list(calc_dict.keys())

        # paraDict is the parameter dictionary read from parameters.json
        eleRootPath = os.path.join(dopeRootPath, str(int(paraDict[element]["AtomicNumber"])) + "-" + element)

        # if overwrite is True, then will use an emptyEleRes to overwrite the eleSumDict
        if overwrite:
            eleSumDict = deepcopy(emptyEleRes)
            overwrite = False

        # if summaryJson is None, then use the emptyEleRes as the template
        # the eleSumDict will only be loaded once 
        eleSummaryJsonPath = os.path.join(eleRootPath, eleSummaryJson)

        if not os.path.isfile(eleSummaryJsonPath) and helpTag:
            eleSumDict = deepcopy(emptyEleRes)
            helpTag = False
        elif eleSumDict is None:
            # because in the end, we will save the results json in every element directory
            # so it doesn't matter which results json we load
            eleSumDict = loadfn(eleSummaryJsonPath)


        # loop over calculations and take the informations 
        for calc in calc_lists:
            calcRootPath = os.path.join(eleRootPath, str(calcs[calc]))

            if calc in [0, 1, 2, 3, 4, 5]:

                print(f'Extracting the {calc_dict[dopsite][calcs[calc]]} of {element} in {dopsite} doping site')
                
                os.chdir(calcRootPath)
                # get energy from OSZICAR
                energy = extract_energy_osz("OSZICAR")
                # get area from CONTCAR (if CONTCAR is empty, use POSCAR instead, but will raise a warning containing the current path)
                if os.path.getsize("CONTCAR") == 0:
                    area = get_area("POSCAR")
                    warnings.warn("CONTCAR is empty, use POSCAR instead, in path: " + calcRootPath)
                else:
                    area = get_area("CONTCAR")
                # calculate the energy per area
                energyPerArea = energy/area
                # store the energy, area, energyPerArea in the eleSumDict
                # the key name can be found in the calc_dict
                eleSumDict[calc_dict[dopsite][calcs[calc]]]["energy"] = energy
                eleSumDict[calc_dict[dopsite][calcs[calc]]]["area"] = area
                eleSumDict[calc_dict[dopsite][calcs[calc]]]["energyPerArea"] = energyPerArea
                eleSumDict[calc_dict[dopsite][calcs[calc]]]["updateTime"] = current_time()
            elif calc in [6, 7, 8, 9]:
                if os.path.exists(calcRootPath):
                    
                    print(f'Extracting the {calc_dict[dopsite][calcs[calc]]} of {element} in {dopsite} doping site')

                    os.chdir(calcRootPath)
                    if calc in [6, 7]:
                        eleSumDict[calc_dict[dopsite][calcs[calc]]] = extract_tensile()
                    elif calc in [8, 9]:
                        eleSumDict[calc_dict[dopsite][calcs[calc]]] = extract_RGS()
                else:
                    warnings.warn("The path: " + calcRootPath + " doesn't exist")
                    eleSumDict[calc_dict[dopsite][calcs[calc]]] = {}
        
        os.chdir(rootPath)
    
    eleSumDict = calc_output_properties(eleSumDict=eleSumDict, ele=element)
    # will dump the json into all folders in the end:
    for iii in whole_doping_list:
        eleRootPath = os.path.join(os.path.abspath(doping_sites_dirs_dict[iii]), str(int(paraDict[element]["AtomicNumber"])) + "-" + element)

        # if summaryJson is None, then use the emptyEleRes as the template
        eleSummaryJsonPath = os.path.join(eleRootPath, eleSummaryJson)
        json.dump(eleSumDict, open(eleSummaryJsonPath, "w"), indent=4)

    return eleSumDict


# def a function, given the eleSumDict, calculate the corresponding output properties
# the reference comes from the undoped system.
def calc_output_properties(eleSumDict, ele):
    ener_GB_ref = -242.47812
    ener_realBulk_ref = -244.41236
    eleSumDict["outputProperties"]["updateTime"] = current_time()

    for jjj in ["Li", "TM"]:
        # 1. grain boundary formation energy (GBFE), which is defined as the difference between the energy of the GB and the energy of the GB bulk structure:
        #    GBFE[J/m2] = 1/2*(ener_GB/area_GB - ener_gbBulk/area_gbBulk)
        if "GB_" + jjj in eleSumDict.keys() and "gbBulk_" + jjj in eleSumDict.keys():
            GBFE = 1/2 * eV2J * 1E20 * (eleSumDict["GB_" + jjj]["energyPerArea"] - eleSumDict["gbBulk_" + jjj]["energyPerArea"])
            eleSumDict["outputProperties"]["GBFE_" + jjj] = GBFE
        else:
            warnings.warn("GBFE_" + jjj + f' for {ele} ' + " can't be calculated")
            eleSumDict["outputProperties"]["GBFE_" + jjj] = "Not applicable"

        # 2. solute segregation energy (SSE) at GB, which is defined as the difference between the energy of the GB and the energy of the bulk structure:
        #    SSE[eV/atom]  = 1/2*( (ener_GB + 2*ener_realBulk_ref) - (ener_GB_ref + 2*ener_realBulk) )
        if "GB_" + jjj in eleSumDict.keys() and "realBulk_" + jjj in eleSumDict.keys():
            SSE_GB = 1/2 * ( (eleSumDict["GB_" + jjj]["energy"] + 2*ener_realBulk_ref) - (ener_GB_ref + 2*eleSumDict["realBulk_" + jjj]["energy"]) )
            eleSumDict["outputProperties"]["SSE_" + jjj] = SSE_GB
        else:
            warnings.warn("SSE_" + jjj + f' for {ele} ' + " can't be calculated")
            eleSumDict["outputProperties"]["SSE_GB_" + jjj] = "Not applicable"

        # 3. SSE at neighbor doping sites, which is defined as the difference between the energy of the neighbor doped GB and the energy of the bulk structure:
        #    SSE_neigh[eV/atom]  = 1/4*( (ener_GB_neigh + 4*ener_realBulk_ref) - (ener_GB_ref + 4*ener_realBulk) )
        if "GB_neigh_" + jjj in eleSumDict.keys() and "realBulk_" + jjj in eleSumDict.keys():
            SSE_neigh = 1/4 * ( (eleSumDict["GB_neigh_" + jjj]["energy"] + 4*ener_realBulk_ref) - (ener_GB_ref + 4*eleSumDict["realBulk_" + jjj]["energy"]) )
            eleSumDict["outputProperties"]["SSE_neigh_" + jjj] = SSE_neigh
        else:
            warnings.warn("SSE_neigh_" + jjj + f' for {ele} ' + " can't be calculated")
            eleSumDict["outputProperties"]["SSE_neigh_" + jjj] = "Not applicable"

        # 4. seperation work (SW), which is defined as the difference between the energy of the GB and the energy of the surface slab:
        #    SW[J/m2] = 1/2*(ener_GB/area_GB - 2*ener_Surf/area_Surf)
        if "GB_" + jjj in eleSumDict.keys() and "Surf_" + jjj in eleSumDict.keys():
            SW = 1/2 * eV2J * 1E20 * (2*eleSumDict["Surf_" + jjj]["energyPerArea"] - eleSumDict["GB_" + jjj]["energyPerArea"])
            eleSumDict["outputProperties"]["SW_" + jjj] = SW
        else:
            warnings.warn("SW_" + jjj + f' for {ele} ' + " can't be calculated")
            eleSumDict["outputProperties"]["SW_" + jjj] = "Not applicable"

        # 5. SW at neighbor doping sites, which is defined as the difference between the energy of the neighbor doped GB and the energy of the surface slab:
        #    SW_neigh[J/m2] = 1/2*(ener_GB_neigh/area_GB_neigh - 2*ener_Surf_neigh/area_Surf_neigh)
        if "GB_neigh_" + jjj in eleSumDict.keys() and "Surf_neigh_" + jjj in eleSumDict.keys():
            SW_neigh = 1/2 * eV2J * 1E20 * (2*eleSumDict["Surf_neigh_" + jjj]["energyPerArea"] - eleSumDict["GB_neigh_" + jjj]["energyPerArea"])
            eleSumDict["outputProperties"]["SW_neigh_" + jjj] = SW_neigh
        else:
            warnings.warn("SW_neigh_" + jjj + f' for {ele} ' + " can't be calculated")
            eleSumDict["outputProperties"]["SW_neigh_" + jjj] = "Not applicable"
    
    return eleSumDict

# main part of this script
if __name__ == "__main__":
    rootPath = os.getcwd()

    # the overall summary will be stored in the allSum.json under each doping site root directory
    # the allSum.json will be updated every time the script is run
    # if the allSum.json exists, then load it
    # as the allSum.json in each doping site root directory is same, so any one of them can be used
    sumJsonPath = os.path.join(os.path.abspath(doping_sites_dirs_dict["GB"]), "allSum.json")
    if os.path.isfile(sumJsonPath):
        allSumDict = loadfn(sumJsonPath)
    else:
        # default template of the allSum.json
        allSumDict = {"updateTime": current_time()}

    # loop over elements list
    for ele in element_lists:
        eleSumDict = extract_all_info(ele, calc_lists, doping_sites, overwrite=False)
        
        ele_index = str(int(paraDict[ele]["AtomicNumber"])) + "-" + ele
        allSumDict[ele_index] = eleSumDict
    
    # set the updating time
    allSumDict["updateTime"] = current_time()

    # extract the output properties from the allSum.json and store them in a dictionary
    sumDictForExcel = {}
    for ele in allSumDict.keys():
        if ele != "updateTime":
            sumDictForExcel[ele] = deepcopy(allSumDict[ele]["outputProperties"])
    sumDfForExcel = pd.DataFrame(sumDictForExcel).T

    # after extracting all the information, the allSum.json will be updated and stored in the root directory of each doping site
    whole_doping_list = list(calc_dict.keys())
    for iii in whole_doping_list:
        dopeRootPath = os.path.abspath(doping_sites_dirs_dict[iii])
        os.chdir(dopeRootPath)
        json.dump(allSumDict, open("allSum.json", "w"), indent=4)
        if os.path.isfile("allSum.xlsx"):
            os.remove("allSum.xlsx")
        sumDfForExcel.to_excel("allSum.xlsx")
        os.chdir(rootPath)