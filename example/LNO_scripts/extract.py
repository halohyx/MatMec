import os
import shutil
import pandas as pd
import numpy as np
from matmec.core import Latt
from monty.serialization import loadfn
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from glob import glob
from copy import deepcopy

element_lists = ["Al"]
calc_lists = [0,1,2,3,4,5]
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

paraDict = loadfn("../parameters.json")

emptyResults = {
    calc0: {"energy": 0.0, "area": 0.0, "energyPerA": 0.0},
    calc1: {"energy": 0.0, "area": 0.0, "energyPerA": 0.0},
    calc2: {"energy": 0.0, "area": 0.0, "energyPerA": 0.0},
    calc3: {"energy": 0.0, "area": 0.0, "energyPerA": 0.0},
    calc4: {"energy": 0.0, "area": 0.0, "energyPerA": 0.0},
    calc5: {"energy": 0.0, "area": 0.0, "energyPerA": 0.0},
    "LiSite-GBForm": 0.0,
    "LiSite-SepWork": 0.0,
    "TMSite-GBForm": 0.0,
    "TMSite-SepWork": 0.0,
    calc6: {},
    calc7: {},
    calc8: {},
    calc9: {},
}

resultsFile = "results.json"
summaryFile = "summary.json"

# times the 1E20 which comes from the 1E-20 of area
eV2J_ = 16.02176565

if not os.path.isfile(summaryFile):
    summary = {}
    for ele in paraDict.keys():
        eleKey = str(paraDict[ele]['AtomicNumber']) + '-' +  ele
        summary[eleKey] = {}
else:
    summary = loadfn(summaryFile)

results = emptyResults

def extract_energy(OUTCAR):
    with open(OUTCAR, 'r') as f:
        for line in f.readlines():
            if 'free  energy   TOTEN' in line:
                # print(line)
                energy = float(line.split()[-2])
                # print(energy)
    return energy

def getEnergyFromOSZICAR(oszicar: str = "OSZICAR"):
    with open(oszicar) as f:
        line = f.readline()
        while line:
            if len(line.split("F=")) > 1:
                energy = float(line.strip().split()[2])
            line = f.readline()
    return energy

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

def extract_stress(OUTCAR):
    '''
    Extract stress from OUTCAR
    '''
    with open(OUTCAR, 'r') as f:
        for line in f.readlines():
            if 'in kB' in line:
                # print(line)
                stress = np.array(line.split()[2:], dtype=float)  

    return stress

def extract_all(OUTCAR):
    '''
    Extract all information from OUTCAR, including energy, volume and stress
    '''
    with open(OUTCAR, 'r') as f:
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

def get_area(POSCAR):
    '''
    Get area from one POSCAR style structure file
    '''
    # unit in A2
    tmp = Latt.read_from_poscar(POSCAR)
    return np.linalg.norm(np.cross(tmp.cell.a, tmp.cell.b))

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

def cubic_spline(x: list, y: list):
    '''
    Returns the cubic spline interpolate, and the corresponding difference at point xi
    '''
    from scipy.interpolate import CubicSpline

    x = np.array(x)
    y = np.array(y)
    cs = CubicSpline(x, y)

    return cs, cs(x, 1)

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

root = os.getcwd()
for ele in element_lists:
    results = deepcopy(emptyResults)
    eleDir = str(int(paraDict[ele]["AtomicNumber"])) + "-" + ele
    os.chdir(eleDir)
    eleRoot = os.getcwd()
    for calc in calc_lists:
        if calc in [0, 1, 2, 3, 4, 5]:
            # extract normal results
            os.chdir(calcs[calc])
            results[calcs[calc]]["energy"] = getEnergyFromOSZICAR()
            results[calcs[calc]]["area"] = get_area("CONTCAR")
            results[calcs[calc]]["energyPerA"] = results[calcs[calc]]["energy"]/results[calcs[calc]]["area"]
            print("Extracting %s of %s" % (calcs[calc], eleDir))
            os.chdir(eleRoot)

        # extract tensile results
        elif calc in [6, 7]:
            os.chdir(calcs[calc])
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
            
            # results[calcs[calc]] = tmpDict
            
            print("Extracting %s of %s" % (calcs[calc], ele))
            os.chdir(eleRoot)
        
        # extract RGS
        elif calc in [8, 9]:
            os.chdir(calcs[calc])
            RGSResultsList = glob("./*/*/*/OSZICAR", recursive=True)
            displist = []
            energylist = []
            area = get_area("POSCAR")
            print(area)
            for ii in RGSResultsList:
                disp = ii.split('/')[-2].split("_")[-1]
                energy = getEnergyFromOSZICAR(ii)
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

            os.chdir(eleRoot)
       
    # postProcess
    if 0 in calc_lists and 1 in calc_lists and 2 in calc_lists:
        # BUG! The GBForm is clearly a bug. Should deduce some new equations later.
        results["LiSite-GBForm"] = eV2J_*(results[calcs[0]]["energy"]/-results[calcs[1]]["energy"])/(2*results[calcs[0]]["area"])
        results["LiSite-SepWork"] = eV2J_*(2*results[calcs[2]]["energyPerA"]-results[calcs[0]]["energyPerA"])/2
    if 3 in calc_lists and 4 in calc_lists and 5 in calc_lists:
        results["TMSite-GBForm"] = eV2J_*(results[calcs[3]]["energy"]-results[calcs[4]]["energy"])/(2*results[calcs[3]]["area"])
        results["TMSite-SepWork"] = eV2J_*(2*results[calcs[5]]["energyPerA"]-results[calcs[3]]["energyPerA"])/2
    
    with open(resultsFile, 'w') as fp:
        fp.write(json.dumps(results, indent=4))
    summary[eleDir] = results
    os.chdir(root)

# write summarization finally
with open(summaryFile, 'w') as fp:
    fp.write(json.dumps(summary, indent=4))