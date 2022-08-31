#!/home/halohyx/app/anaconda3/bin/python
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from pymatgen.electronic_structure.core import Spin, OrbitalType
from pymatgen.io.vasp import Vasprun
import numpy as np


colormap =  np.array([[244, 124, 124], #pink red
            [247, 244, 139], #light yellow
            [161, 222, 147], #light green
            [112, 161, 215], #light blue
            [184, 91, 63],   #brown
            [174, 76, 207],  #purple
            [18, 0, 120], #dark blue
            ])/255




def draw_RGBALine(ax, k, e, color_distribution, alpha = 1.):
    points = np.array([k, e]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)

    ax = ax
    if len(np.array(color_distribution).shape) == 1:
        lc = LineCollection(segments, colors = color_distribution, linewidths=2)
    else:
        newdistribution = [ 0.5*(color_distribution[i] + color_distribution[i+1]) for i in range(len(color_distribution) - 1) ]
        newColorMap = np.matmul(newdistribution, colormap[:len(color_distribution[0])])
        a = np.ones((len(newColorMap), 1), float) * alpha
        lc = LineCollection(segments, colors = np.concatenate((newColorMap, a), axis = 1) , linewidths=2)
    ax.add_collection(lc)


# read vasp output file in current folder 
run = Vasprun('./vasprun.xml', parse_projected_eigen=True)
# DOS
dos = run.complete_dos
# BAND Structure
bands = run.get_band_structure('./KPOINTS', line_mode=True ,efermi=dos.efermi)
# Projected BandStructure on each element
pbands = bands.get_projection_on_elements()

# write a BAND_GAP.dat file to save the bandgap and bandgap type information
with open('BAND_GAP.dat', 'wt') as f:
    s = ''
    bandgap = bands.get_band_gap()
    s += r'BAND GAP: %s      ' % bandgap['energy']
    if bandgap['direct']:
        s += r'BAND GAP TYPE: %s ' % 'Direct'
    else:
        s += r'BAND GAP TYPE: %s ' % 'Indirect'
    print(s, file=f)


# basic information of this band structure file like NBAND, number of KPOINTS, 
# elements list, number of elements, labels of KPOINTS, initialize the contribution list of each element
# k is the x axis of KPOINTS
nbands = bands.nb_bands
nkpoints = len(bands.kpoints)
nelements = len(pbands[Spin.up][0][0].keys())
elements_list = list(pbands[Spin.up][0][0].keys())
contribution = np.zeros((nbands, nkpoints, nelements))
labels = bands.branches[0]['name'].split('-')
for i in range(1, len(bands.branches)):
    if labels[-1] == bands.branches[i]['name'].split('-')[0]:
        labels.append(bands.branches[i]['name'].split('-')[-1])
    else:
        labels[-1] = '%s,%s' % (labels[-1], bands.branches[i]['name'].split('-')[0])
        labels.append(bands.branches[i]['name'].split('-')[-1])
labels = [ r'$%s$' % i for i in labels ]
k = range(nkpoints)


#----------SET MATPLOTLIB-------------#

# set MATPLOTLIB default style
font = {'family': 'serif', 'size': 20}
plt.rc('font', **font)


# Define two ax using GridSpec, ax1 is for BAND structure and ax2 is for DOS structure
gs = GridSpec(1, 2, width_ratios=[2, 1])
fig = plt.figure(figsize=(11.69, 8.27))
name = input('Please input the name of the system:  ')
fig.suptitle("Band Diagram of %s" % name)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# Calculate the maximum energy and minimum energy to set the limit of Y axis
emax = -1E100
emin = 1E100
for spin in bands.bands.keys():
    for band_NO in range(nbands):
        emax = max(emax, max(bands.bands[spin][band_NO]))
        emin = min(emin, min(bands.bands[spin][band_NO]))
emax -= bands.efermi - 1
emin -= bands.efermi + 1
lowerLimit = input('You want to set the lower limit? Y/N')
if lowerLimit[0] in ['Y', 'y']:
    emin = eval(input('Please set the lower limit:   '))
else:
    pass

ax1.set_ylim(emin, emax)
ax2.set_ylim(emin, emax)


#-------BAND-------------#

# iterate between projected BANDs to generate an array called contribution which saved the 
# information of the contribution of each element at each KPOINT of each BAND
for spin in pbands.keys():
    for band_NO in range(nbands):
        for kpoints_NO in range(nkpoints):
            for element_num, element_key in zip(range(nelements), pbands[spin][band_NO][kpoints_NO]):
                # print(pbands[spin][band_NO][kpoints_NO][element_key])
                contribution[band_NO, kpoints_NO, element_num] = pbands[spin][band_NO][kpoints_NO][element_key]
            # normalize
            contribution_summation = sum(contribution[band_NO, kpoints_NO])
            if contribution_summation != 0:
                for i in range(len(contribution[band_NO, kpoints_NO])):
                    contribution[band_NO, kpoints_NO, i] = contribution[band_NO, kpoints_NO, i]/contribution_summation
            del contribution_summation

# Draw BAND structure
for i in range(len(contribution)):
    # draw_RGBALine(ax1, k, bands.bands[Spin.up][i] - bands.efermi, contribution[i])
    draw_RGBALine(ax1, k, bands.bands[Spin.up][i] - bands.efermi, color_distribution= colormap[3])


# Draw KPOINTS labels and specify the grids with those labels
nlabels = len(labels)
step = (nkpoints - 1) / (nlabels - 1)
for i, lab in enumerate(labels):
    ax1.vlines(i * step, emin, emax, "k")
ax1.set_xticks([i * step for i in range(nlabels)])
ax1.set_xticklabels(labels)
ax1.set_xlim(0, nkpoints-1)

# set the fermi level line
ax1.hlines(y=0, xmin=0, xmax=len(bands.kpoints), color="k", lw=2, zorder = 3, alpha = 0.6)
ax1.grid()

# set X axis title and Y axis title
ax1.set_xlabel("k-points")
ax1.set_ylabel(r"$E - E_f$   /   eV")


#--------------DOS---------------#
ax2.set_yticklabels([])
ax2.grid()
ax2.set_xlim(1e-6, max(run.tdos.densities[Spin.up])+1)
ax2.set_xticklabels([])
ax2.hlines(y=0, xmin=0, xmax=max(run.tdos.densities[Spin.up])+1, color="k", lw=2, zorder = 3, alpha = 0.6)
ax2.set_xlabel("Density of States", labelpad=28)
plt.subplots_adjust(wspace=0)


# draw orbitals contribution
splitDos = run.complete_dos.get_spd_dos()
orbitals = []
for i in splitDos.keys():
    orbitals.append(str(i))
for i, orbital in enumerate(splitDos.keys()):
    ax2.plot(splitDos[orbital].densities[Spin.up],
             run.tdos.energies - run.efermi,
             label="%s" % orbitals[i], lw=2) 

# total DOS
ax2.fill_between(run.tdos.densities[Spin.up],
                    0,
                    run.tdos.energies - run.efermi,
                    color=(0.7, 0.7, 0.7),
                    facecolor=(0.7, 0.7, 0.7))

ax2.plot(run.tdos.densities[Spin.up],
            run.tdos.energies - run.efermi,
            color=(0.3, 0.3, 0.3),
            label="total DOS")


ax2.legend(fancybox=True, shadow=False, prop={'size': 18})
fig.savefig('%s.jpeg' % name)