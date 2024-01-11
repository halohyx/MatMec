"""
Make a NEB run from two images
"""

import os
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar
from pymatgen.analysis.transition_state import NEBAnalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)

def make_neb(ini_dir, fin_dir, image_num):
    """
    Make a NEB run from two images
    """
    # get the absolute path of the initial and final structures
    ini_structure = os.path.abspath(os.path.join(ini_dir, "CONTCAR"))
    fin_structure = os.path.abspath(os.path.join(fin_dir, "CONTCAR"))

    # check if the initial and final structures have converged
    # use pymatgen to check the outcar
    ini_outcar_path = os.path.join(ini_dir, "OUTCAR")
    fin_outcar_path = os.path.join(fin_dir, "OUTCAR")

    ini_convergence = Vasprun(os.path.join(ini_dir, "vasprun.xml")).converged
    if not ini_convergence:
        warnings.warn(f'Initial structure in {ini_dir} has not converged')

    fin_convergence = Vasprun(os.path.join(fin_dir, "vasprun.xml")).converged
    if not fin_convergence:
        warnings.warn(f'Final structure in {fin_dir} has not converged')

    # make the neb working folder and subfolders
    neb_dir = make_dir(f'{ini_dir}_{fin_dir}_neb')

    os.chdir(neb_dir)

    # run the nebmake.pl command, remember to replace this with the absolute path
    os.system(f'/home/halohyx/app/vtst/vtstscripts-1033/nebmake.pl {ini_structure} {fin_structure} {image_num}')

    # copy the OUTCARs to the neb folder
    os.system(f'cp {ini_outcar_path} {neb_dir}/00/OUTCAR')
    os.system(f'cp {fin_outcar_path} {neb_dir}/{image_num+1:02d}/OUTCAR')


def make_neb_incar(images = None, **kwargs):
    '''
    Make a NEB INCAR, already set the default values
    Make sure to atleast give the IMAGES value
    '''
    incar = Incar.from_dict({
        'SYSTEM': 'neb',
        "ISTART": 0,
        "ICHARG": 2,
        "NCORE": 4,
        "LREAL": "Auto",

        "IBRION": 3,
        "POTIM": 0,
        "NSW": 100,

        "PREC": "Normal",
        "EDIFF": 1e-5,
        "EDIFFG": -0.01,
        "ENCUT": 400,

        "ALGO": "Normal",
        "ISMEAR": 0,
        "SIGMA": 0.10,

        "LWAVE": False,
        "LCHARG": False,

        "ICHAIN": 0,
        "LCLIMB": True,
        "SPRING": -5,
        "IOPT": 2,
    })

    incar['IMAGES'] = int(images)

    for key, value in kwargs.items():
        incar[key] = value

    return incar

def neb_analysis(neb_dir):
    '''
    Ultilize the NEBAnalysis class to analyze the NEB run
    '''
    neb_analysis = NEBAnalysis.from_dir(neb_dir)

    plt.figure(figsize=(12, 8))

    ax = plt.gca()

    x = np.arange(0, np.max(neb_analysis.r), 0.01)
    y = neb_analysis.spline(x) * 1000
    relative_energies = neb_analysis.energies - neb_analysis.energies[0]

    scale = 1 / neb_analysis.r[-1]

    # plot the spline
    ax.plot(
            x * scale,
            y,
            "k--",
            linewidth=3,
            zorder = 1
            )

    # plot the calculated energies
    ax.scatter(
            neb_analysis.r * scale,
            relative_energies * 1000,
            color = "r",
            s = 200,
            zorder = 2   
    )

    # set the titles of x and y axis
    label_font = {"fontname": "Calibri", "size": 20, "color": [0.2, 0.2, 0.2], "weight": "bold", "style": "italic"}

    ax.set_xlabel("Reaction coordinate", fontdict=label_font)
    ax.set_ylabel("Energy (meV)", fontdict=label_font)

    # add minor ticks
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # set the Y axis limits
    ax.set_ylim((np.min(y) * 1.1, np.max(y) * 1.2))

    # set the tick parameters
    ax.tick_params(axis="both", which="major",
                    length = 8, width = 2, 
                    labelsize=14, color = [0.2, 0.2, 0.2])

    # add grids
    ax.grid(which="both", axis="both", linestyle="--", color=[0.9, 0.9, 0.9])

    plt.savefig(f'{neb_dir}/neb_analysis.png', dpi=300, bbox_inches='tight')

    return neb_analysis
