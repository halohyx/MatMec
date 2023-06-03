from matmec.core import Latt, Atom, Cell
from matmec.tool.latt_tool import get_distances
import os
import numpy as np
from monty.serialization import loadfn
import shutil
from pymatgen.io.vasp.inputs import Incar

#-------The following 4 variables are what you may want to change-------#
# the elements you will calculate
element_lists = ["Eu"]

# the calculation list
calc_lists = [0, 1, 2, 3, 4, 5]

# whether to skip the existed calculated ones
skip_calculated = False

# the suffix of the backup folder
bakName = None
#------------------------------------------------------------------------#

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

paraDict = loadfn("../parameters.json")
paraDict

# define the displacements in RGS calculation
RGSdis  = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4, 5, 6]
# RGSdis = [0.0]
calcs = [calc0, calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9]
dopeSites = ["Li", "Li", "Li", "TM", "TM", "TM"]
calcTypes = ["GB", "Bulk", "Surf", "GB", "Bulk", "Surf"]

#---------------------------------------------------------------------#
#Definition of INCAR
#---------------------------------------------------------------------#
class hyxincar:
    def __init__(self, dopeEle: str, dopeSite = "Li", calcType="GB"):
        baseTemp = '''
ISTART = 0
ICHARG = 2
NCORE = 4
NSIM = 4              # how many bands parellel in the RMM-DIIS algorithm
LREAL = Auto

# ionic relaxation
NSW     = 500       # number of ionic steps
IBRION  = 2         # 2=conjucate gradient, 1=Newton like
ISIF    = 3         # 3=relax everything, 2=relax ions only, 4=keep volume fixed

# POTIM = 0.2
ADDGRID = .FALSE.
ISPIN = 2 #Is the system magenetic?

# precision parameters
PREC    = Normal  # precision normal, accurate
EDIFF   =  5E-6     # 1E-3 very low precision for pre-relaxation, use 1E-5 next
# EDIFFG  =  -2E-2     # usually: 10 * EDIFF
NELMIN  = 6
NELM  = 150

# electronic relaxation
ALGO    =  Normal
ENCUT   =  600 # cutoff energy
ISMEAR  =  0      # -5 = tetraedon, 1..N = Methfessel, metal=1, for insulator, set it to be 0
SIGMA   =  0.05  # default 0.2

# output options
LWAVE   = .FALSE.   # write or don't write WAVECAR
LCHARG  = .FALSE.    # write or don't write CHG and CHGCAR
# METAGGA = SCAN

IVDW    = 11
LORBIT  = 11

# AMIX     = 0.02
# BMIX     = 0.0001
# AMIX_MAG = 0.8
# BMIX_MAG = 0.0001 ! almost zero, but 0 will crash some versions

# +U
# PBE + U
LDAU = True         # switch on L(S)DA + U
LDAUTYPE = 2        # defualt
'''
        Uorb = paraDict[dopeEle]["Uorb"]
        HubbardU = paraDict[dopeEle]["HubbardU"]
        Mag = paraDict[dopeEle]["Mag"]

        baseTemp += "LDAUL = -1 %d 2 -1 \n" % Uorb
        baseTemp += "LDAUU = 0.0 %.2f 7.17 0.0\n" % HubbardU
        baseTemp += "LDAUJ = 0.0 %.2f 0.0 0.0\n" % 0.0
        if calcType == "Surf":
            if dopeSite == "Li":
                baseTemp += "MAGMOM = 4*0.0 2*0.0 6*1.8 12*0.0\n" # % Mag
            elif dopeSite == "TM":
                baseTemp += "MAGMOM = 6*0.0 2*0.0 4*1.8 12*0.0\n" # % Mag
        elif calcType == "Bulk":
            if dopeSite == "Li":
                baseTemp += "MAGMOM = 10*0.0 2*0.0 12*1.8 24*0.0\n" # % Mag
            elif dopeSite == "TM":
                baseTemp += "MAGMOM = 12*0.0 2*0.0 10*1.8 24*0.0\n" # % Mag
        elif calcType == "GB":
            if dopeSite == "Li":
                baseTemp += "MAGMOM = 8*0.0 4*0.0 12*1.8 24*0.0\n" # % Mag
            elif dopeSite == "TM":
                baseTemp += "MAGMOM = 12*0.0 4*0.0 8*1.8 24*0.0\n" # % Mag
        if Uorb == 3:
            baseTemp += "LMAXMIX = 6\n"
        else:
            baseTemp += "LMAXMIX = 4\n"
        self.incar = baseTemp

    def toIncar(self):
        return Incar.from_string(self.incar)

    def __repr__(self):
        return self.incar

#---------------------------------------------------------------------#
#Definition of POSCAR
#---------------------------------------------------------------------#
class poscar:

    GB_temp = '''012 GB_neighbor {elements}                                                                      
   1.00000000000000     
     2.8320244804670498    0.0000000000000000    0.0000000000000000
     0.0000000000000000    4.9661525702770062    0.0000000000000000
     0.0000000000000000    0.0000000000000000   28.2733142166249891
   {elements}
   {eleCounts}
Direct
  0.5000000000000000  0.1740670383871696  0.5824767237414014
  0.5000000000000000  0.8407758400192291  0.4180037129277364
  0.0000000000000000  0.0068114004715469  0.4999326927548265
  0.5000000000000000  0.1740670383871696  0.9175232762585986
  0.5000000000000000  0.8407758400192291  0.0819961980722667
  0.0000000000000000  0.0068114004715469  0.0000673072451731
  0.5000000000000000  0.5224090724964408  0.2500000000000000
  0.5000000000000000  0.5020256176969896  0.7500000000000000
  0.0000000000000000  0.6878375488504731  0.1664710084676044
  0.0000000000000000  0.3331774187376230  0.8338385380296397
  0.0000000000000000  0.3331774187376230  0.6661614029703661
  0.0000000000000000  0.6878375488504731  0.3335290505323893
  0.5000000000000000  0.8287372040067840  0.6650096832884113
  0.5000000000000000  0.1888785847475109  0.1649249631968402
  0.5000000000000000  0.8287372040067839  0.8349902577115945
  0.5000000000000000  0.1888785847475109  0.3350750958031539
  0.0000000000000000  0.0173077592560106  0.2500000000000000
  0.0000000000000000  0.0078050688068384  0.7500000000000000
  0.0000000000000000  0.3355296965938017  0.4177165948044156
  0.5000000000000000  0.5072035083256045  0.4995791550951738
  0.0000000000000000  0.6777615729331551  0.5816626970501420
  0.0000000000000000  0.3355296965938017  0.0822833161955871
  0.5000000000000000  0.5072035083256045  0.0004208449048191
  0.0000000000000000  0.6777615729331551  0.9183373029498580
  0.5000000000000000  0.8843168379544079  0.2962342297154113
  0.0000000000000000  0.0399480049761079  0.3758521541019120
  0.0000000000000000  0.3444633364732655  0.2966820526405344
  0.5000000000000000  0.1780007721802284  0.4567932850821870
  0.0000000000000000  0.3752117235637729  0.5423885764309098
  0.5000000000000000  0.5267152560384103  0.6201545697268777
  0.0000000000000000  0.6743382690718961  0.7027496637857421
  0.5000000000000000  0.4853222506164060  0.3772444802375582
  0.0000000000000000  0.9722090721153716  0.6228851513054964
  0.5000000000000000  0.8342086495645114  0.5430455354577340
  0.5000000000000000  0.1397362159733825  0.7038604330825258
  0.0000000000000000  0.6421659422712087  0.4565755050668685
  0.5000000000000000  0.8843168379544079  0.2037657402845938
  0.0000000000000000  0.0399480049761079  0.1241479348980849
  0.0000000000000000  0.3444633364732655  0.2033179183594592
  0.5000000000000000  0.1780007721802283  0.0432067439178116
  0.0000000000000000  0.3752117235637729  0.9576113635690924
  0.5000000000000000  0.5267152560384103  0.8798454302731223
  0.0000000000000000  0.6743382690718961  0.7972503952142592
  0.5000000000000000  0.4853222506164060  0.1227554607624407
  0.0000000000000000  0.9722090721153716  0.8771148486945036
  0.5000000000000000  0.8342086495645114  0.9569544645422660
  0.5000000000000000  0.1397362159733825  0.7961396269174721
  0.0000000000000000  0.6421659422712087  0.0434244949331313
    '''

    bulk_temp = '''GB_center {elements}                                                                                                                                                                                          
   1.00000000000000     
     2.8320244804670498    0.0000000000000000    0.0000000000000000
     0.0000000000000000    4.9661525702770062    0.0000000000000000
     0.0000000000000000    0.0000000000000000   28.2733142166249891
   {elements}
   {eleCounts}
Direct
  0.5000000000000000  0.1740670383871696  0.5824767237414014
  0.5000000000000000  0.8407758400192291  0.4180037129277364
  0.5000000000000000  0.1740670383871696  0.9175232762585986
  0.5000000000000000  0.8407758400192291  0.0819961980722667
  0.5000000000000000  0.5224090724964408  0.2500000000000000
  0.5000000000000000  0.5020256176969896  0.7500000000000000
  0.0000000000000000  0.6878375488504731  0.1664710084676044
  0.0000000000000000  0.3331774187376230  0.8338385380296397
  0.0000000000000000  0.3331774187376230  0.6661614029703661
  0.0000000000000000  0.6878375488504731  0.3335290505323893
  0.0000000000000000  0.0068114004715469  0.4999326927548265
  0.0000000000000000  0.0068114004715469  0.000067307245173
  0.5000000000000000  0.5072035083256045  0.4995791550951738
  0.5000000000000000  0.5072035083256045  0.0004208449048191
  0.5000000000000000  0.8287372040067840  0.6650096832884113
  0.5000000000000000  0.1888785847475109  0.1649249631968402
  0.5000000000000000  0.8287372040067839  0.8349902577115945
  0.5000000000000000  0.1888785847475109  0.3350750958031539
  0.0000000000000000  0.0173077592560106  0.2500000000000000
  0.0000000000000000  0.0078050688068384  0.7500000000000000
  0.0000000000000000  0.3355296965938017  0.4177165948044156
  0.0000000000000000  0.6777615729331551  0.5816626970501420
  0.0000000000000000  0.3355296965938017  0.0822833161955871
  0.0000000000000000  0.6777615729331551  0.9183373029498580
  0.5000000000000000  0.8843168379544079  0.2962342297154113
  0.0000000000000000  0.0399480049761079  0.3758521541019120
  0.0000000000000000  0.3444633364732655  0.2966820526405344
  0.5000000000000000  0.1780007721802284  0.4567932850821870
  0.0000000000000000  0.3752117235637729  0.5423885764309098
  0.5000000000000000  0.5267152560384103  0.6201545697268777
  0.0000000000000000  0.6743382690718961  0.7027496637857421
  0.5000000000000000  0.4853222506164060  0.3772444802375582
  0.0000000000000000  0.9722090721153716  0.6228851513054964
  0.5000000000000000  0.8342086495645114  0.5430455354577340
  0.5000000000000000  0.1397362159733825  0.7038604330825258
  0.0000000000000000  0.6421659422712087  0.4565755050668685
  0.5000000000000000  0.8843168379544079  0.2037657402845938
  0.0000000000000000  0.0399480049761079  0.1241479348980849
  0.0000000000000000  0.3444633364732655  0.2033179183594592
  0.5000000000000000  0.1780007721802283  0.0432067439178116
  0.0000000000000000  0.3752117235637729  0.9576113635690924
  0.5000000000000000  0.5267152560384103  0.8798454302731223
  0.0000000000000000  0.6743382690718961  0.7972503952142592
  0.5000000000000000  0.4853222506164060  0.1227554607624407
  0.0000000000000000  0.9722090721153716  0.8771148486945036
  0.5000000000000000  0.8342086495645114  0.9569544645422660
  0.5000000000000000  0.1397362159733825  0.7961396269174721
  0.0000000000000000  0.6421659422712087  0.0434244949331313
    '''

    surf_temp = '''Surf_neighbor {elements}                                                                                        
   1.00000000000000     
     2.8712692245305762    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.2039521960427626   -0.0204150793480340
     0.0000000000000000   -0.1116638849032437   27.5440995499922785
   {elements}
   {eleCounts}
Selective dynamics
Direct
  0.5000000000000000  0.5112118178615337  0.5130817459945684   T   T   T
  0.5000000000000000  0.1706045360119662  0.6660008430391683   T   T   T
  0.0000000000000000  0.0142519411899713  0.7497276160058861   T   T   T
  0.0000000000000000  0.6628790868160024  0.9215762800174682   T   T   T
  0.0000000000000000  0.3162019204924219  0.5916413894665272   T   T   T
  0.5000000000000000  0.8246190165973252  0.8342510283304934   T   T   T
  0.5000000000000000  0.8282345364564302  0.5815628255414693   T   T   T
  0.0000000000000000  0.3598822774261137  0.8293946601499433   T   T   T
  0.5000000000000000  0.1723580882491559  0.9110178561685087   T   T   T
  0.5000000000000000  0.5363414311510184  0.7464453677404715   T   T   T
  0.0000000000000000  0.0101936497306862  0.5039267997281516   T   T   T
  0.0000000000000000  0.6744937427176261  0.6631032066571484   T   T   T
  0.5000000000000000  0.9017454432146329  0.9498250306956003   T   T   T
  0.0000000000000000  0.0390877890169270  0.8724575992026320   T   T   T
  0.0000000000000000  0.2883271728286274  0.9477931993574938   T   T   T
  0.5000000000000000  0.2406727258543318  0.7841057262554485   T   T   T
  0.0000000000000000  0.3924713616736610  0.7040876313674495   T   T   T
  0.5000000000000000  0.5127035733797896  0.6221788642052203   T   T   T
  0.0000000000000000  0.6933833396629211  0.5299976635783017   T   T   T
  0.5000000000000000  0.4812808730715520  0.8748806617249477   T   T   T
  0.0000000000000000  0.9723984194656173  0.6233975697707438   T   T   T
  0.5000000000000000  0.8223827127223537  0.7049167171797086   T   T   T
  0.5000000000000000  0.1822843475107204  0.5285552844947295   T   T   T
  0.0000000000000000  0.6852131413866197  0.7873148812375824   T   T   T
    '''

    def __init__(self, dopeEle: str, dopeSite = "Li", calcType="Bulk"):
        
        if calcType == "Surf":
          elements = "Li %s Ni O" % dopeEle
          if dopeSite == "Li":
            eleCounts = "4 2 6 12"
          elif dopeSite == "TM":
            eleCounts = "6 2 4 12"
        elif calcType == "Bulk":
          elements = "Li %s Ni O" % dopeEle
          if dopeSite == "Li":
            eleCounts = "10 2 12 24"
          elif dopeSite == "TM":
            eleCounts = "12 2 10 24"
        elif calcType == "GB":
          elements = "Li %s Ni O" % dopeEle
          if dopeSite == "Li":
            eleCounts = "8 4 12 24"
          elif dopeSite == "TM":
            eleCounts = "12 4 8 24"
        
        if calcType == "Surf":
          self.poscar = self.surf_temp.format(elements=elements, eleCounts=eleCounts)
        elif calcType == "Bulk":
          self.poscar = self.bulk_temp.format(elements=elements, eleCounts=eleCounts)
        elif calcType == "GB":
          self.poscar = self.GB_temp.format(elements=elements, eleCounts=eleCounts)
    
    def to_txt(self):
      return self.poscar

    def __repr__(self) -> str:
      return self.poscar

#---------------------------------------------------------------------#
# Function that call vaspkit to generate KPOINTS and POTCAR
# and submit job
#---------------------------------------------------------------------#
def callVaspkit(if_surf: bool=False):
    vaspkitInput = '''\
102
2
0.04

    '''
    with open('vaspkitInput', 'w') as f:
        f.write(vaspkitInput)
    os.system("vaspkit < vaspkitInput 1>/dev/null")
    if if_surf:
        os.system("touch vasp_ZZ.slurm")
    else:
        os.system("touch vasp_630.slurm")

#---------------------------------------------------------------------#
#Function that call vaspkit to generate KPOINTS and POTCAR
#and submit job
#---------------------------------------------------------------------#

def callVaspkitPOTCAR():
    vaspkitInput = '''\
103
    '''
    if os.path.isfile("POTCAR"):
        os.remove("POTCAR")
    with open('vaspkitInput', 'w') as f:
        f.write(vaspkitInput)
    os.system("vaspkit < vaspkitInput 1>/dev/null")

#---------------------------------------------------------------------#
#Function that change certain tag in INCAR
#---------------------------------------------------------------------#
def sedINCAR(tag, newtag):
    with open('INCAR', 'r') as incar:
        line = incar.readline()
        newINCAR = ''
        while line:
            if tag in line:
                newINCAR += newtag
                newINCAR += "\n"
            else:
                newINCAR += line
            line = incar.readline()
    with open("INCAR", "w") as incar:
        incar.write(newINCAR)

def copyHelper(originPath, newName, overwrite: bool = True):
    if os.path.isfile(newName):
        if overwrite:
            print("Overwrite %s" % newName)
            shutil.copy(originPath, newName)
        else:
            print("***Copy %s to %s failed, %s already exists" % (originPath, newName, newName))
    else:
        shutil.copy(originPath, newName)
            
#---------------------------------------------------------------------#
#Function that generate INCAR and POSCAR
#---------------------------------------------------------------------#
def genInput(ele: str, stage: int):
    if skip_calculated and os.path.isdir(calcs[stage]):
        os.chdir(calcs[stage])
        return calcs[stage]
    else:
        create_path(calcs[stage], back=True, bakName=bakName)
        os.chdir(calcs[stage])
        # this new hyxincar is used from pymatgen.io.vasp.inputs.Incar
        hyxincar(dopeEle=ele, dopeSite=dopeSites[stage], calcType=calcTypes[stage]).toIncar().write_file("INCAR")
        poscarTXT = poscar(dopeEle=ele, dopeSite=dopeSites[stage], calcType=calcTypes[stage]).to_txt()
        with open("POSCAR", 'w') as f:
            f.write(poscarTXT)
        print("Do %s of %s" % (calcs[stage], ele))
        return calcs[stage]

#---------------------------------------------------------------------#
# function that create dirs, copied from DPGEN
#---------------------------------------------------------------------#
def create_path (path,
                 back=False,
                 bakName=None):
    if  path[-1] != "/":
        path += '/'
    if os.path.isdir(path) : 
        if back:
           dirname = os.path.dirname(path)
           counter = 0
           while True :
               if bakName:
                    bk_dirname = dirname + f'.{bakName}'
                    if os.path.isdir(bk_dirname):
                        raise ValueError(f'Backup folder {bk_dirname} alreaddy exist') 
                    else:
                        shutil.move(dirname, bk_dirname)
                        break
               else:
                    bk_dirname = dirname + f'.bk{counter:03d}'
               if not os.path.isdir(bk_dirname) : 
                   shutil.move(dirname, bk_dirname) 
                   break
               counter += 1
           os.makedirs(path)
           return path
        else:
           return path

    os.makedirs (path)
    return path

#---------------------------------------------------------------------#
#Function that do tensile on C direction
#---------------------------------------------------------------------#
def tensile(stage: int):
    # BUG! BUG all here, check before use!
    if stage == 6 or stage == 7:
        os.mkdir(calcs[stage])
        if stage == 6:
            originPath = calcs[0] 
        elif stage == 7:
            originPath = calcs[3]
        originContcar = os.path.abspath(os.path.join(originPath, "CONTCAR"))
        originIncar = os.path.abspath(os.path.join(originPath, "INCAR"))
        originKpoints = os.path.abspath(os.path.join(originPath, "KPOINTS"))
        originPotcar = os.path.abspath(os.path.join(originPath, "POTCAR"))
        os.chdir(calcs[stage])
        copyHelper(originContcar, "POSCAR")
        copyHelper(originIncar, "INCAR")
        incar = Incar.from_file("INCAR")
        incar.update({"IBRION" : 2})
        copyHelper(originKpoints, "KPOINTS")
        copyHelper(originPotcar, "POTCAR")

        # a patch for calculating 28-Ni with correct U value
        if ele == "Ni":
                incar.update({"LDAUJ": "0.0 0.0 0.0",
                              "LDAUL": "-1 2 -1",
                              "LDAUU": "0.0 7.17 0.0"})
                _ = Latt.read_from_poscar()
                _.write_to_poscar("POSCAR")
                callVaspkitPOTCAR()

        incar.write_file("INCAR")
        if not os.path.exists('./1-tensile'):
            os.mkdir('./1-tensile')

        os.system('cp ./POSCAR ./1-tensile/POSCAR')
        # pls change the structure file name
        os.chdir('./1-tensile')

        a = Latt.read_from_poscar('POSCAR')
        a.uniaxial_tensile(2, 1.0, 1.15, 10)
        cal_list = []
        for i in os.listdir():
            if 'Tensile' in i and i.split('.')[-1] == 'vasp':
                cal_list.append(i)

        process_name = 'tensile'

        for i in cal_list:
            serial_number = None
            cwd = os.getcwd()
            if len(i.split('_')) in [3, 4] :
                '''
                BUG Have't been runed after writing this 20221228, pls carefully check before using this Tensile version
                '''
                if not os.path.exists('./%s' % process_name):
                    create_path('./%s' % process_name, back=False)
                serial_number = i.split('Tensile_')[-1].split('.vasp')[0]
                if skip_calculated:
                    continue
                create_path('./%s/%s' % (process_name, serial_number), back=True)
                os.system('cp %s %s/%s/POSCAR' % (i, process_name, serial_number))
                os.system('cp ../INCAR %s/%s' % (process_name, serial_number))
                os.system('cp ../KPOINTS %s/%s' % (process_name, serial_number))
                os.system('cp ../POTCAR %s/%s' % (process_name, serial_number))
                os.chdir('./%s/%s' % (process_name, serial_number))
                print('I am in %s'%(os.getcwd()))
                os.system('touch vasp_tensile.slurm')
                # os.system('touch %s' % '~/bin/vasp_cpu.slurm')
                # pls change the command to submit the script
                os.chdir(cwd)
    else:
        raise ValueError('Dont call me when stage is other than 6 or 7')

def RGS(stage: int, ele: str, disBetwGrains = 5, relax: bool = False):
    '''
    Do the Rigid Grain Shift.
    Args:
        Stage: whether do for Li site or TM site.
        ele: what the doping element is it for determing the GB?
        disBetwGrains: the Distance between two grains.
        relax: do the relaxation of not.
    '''
    if stage == 8 or stage == 9:
        if not os.path.isdir(calcs[stage]):
            os.mkdir(calcs[stage])
        if stage == 8:
            originPath = calcs[0] 
        elif stage == 9:
            originPath = calcs[3]
        originContcar = os.path.abspath(os.path.join(originPath, "CONTCAR"))
        originIncar = os.path.abspath(os.path.join(originPath, "INCAR"))
        originKpoints = os.path.abspath(os.path.join(originPath, "KPOINTS"))
        originPotcar = os.path.abspath(os.path.join(originPath, "POTCAR"))
        os.chdir(calcs[stage])
        copyHelper(originContcar, "POSCAR")
        copyHelper(originIncar, "INCAR")
        copyHelper(originKpoints, "KPOINTS")
        copyHelper(originPotcar, "POTCAR")
        incar = Incar.from_file("INCAR")
        if relax:
            incar.update({"ISIF" : 2, 
                          "ENCUT" : 600, 
                          "PREC" : "Normal",
                          "ADDGRID" : False, 
                          "NCORE" : 4})
        else:
            incar.update({"ISIF" : 2,
                          "NSW" : 0,
                          "IBRION" : -1,
                          "ENCUT" : 600, 
                          "PREC" : "Normal", 
                          "ADDGRID" : False, 
                          "NCORE" : 4,
                          "NELM" : 300})
        
        # a patch for calculating 28-Ni with correct U value
        if ele == "Ni":
                incar.update({"LDAUJ": "0.0 0.0 0.0",
                              "LDAUL": "-1 2 -1",
                              "LDAUU": "0.0 7.17 0.0"})
                _ = Latt.read_from_poscar()
                _.write_to_poscar("POSCAR")
                callVaspkitPOTCAR()
        
        incar.write_file("INCAR")

        if not os.path.exists('./1-RGS'):
            os.mkdir('./1-RGS')
        
        os.chdir("./1-RGS")
        copyHelper(originContcar, "POSCAR")
        # read the POSCAR as input
        a = Latt.read_from_poscar()
    
        # we select the GB boundary as 0.01 above the O atoms connected to the doping element
        poslist = np.array([ i.pos for i in np.append(a.atomlist[ele], a.atomlist["O"])]).reshape(-1, 3)
        # get the xyz minus xyz in an element-wise way
        xyzDist = get_distances(poslist, cell=a.cell)[0]
        # the xyz minus xyz of the 2nd element and 1st element, take Z only
        # find GB1 position
        GB1zList = xyzDist[0][:, 2]
        GB1mask = GB1zList <= 0
        GB1zList[GB1mask] = 1E100
        GB1pos = poslist[np.argmin(GB1zList)][2] + 0.01
        # find GB2 position
        GB2zList = xyzDist[1][:, 2]
        GB2mask = GB2zList <= 0
        GB2zList[GB2mask] = 1E100
        GB2pos = poslist[np.argmin(GB2zList)][2] + 0.01

        # make sure GB1 is the higher one
        if GB1pos < GB2pos:
            temp = float(GB1pos)
            GB1pos = float(GB2pos)
            GB2pos = float(temp)

        '''
        Grain 1 is the one above GB1 and lower than GB2, so we set Grain1Up and Grain1Low
        Grain 2 is the one between GB1 and GB2
        '''
        Grain1Up = np.where(a.poslist[:, 2] > GB1pos)[0]
        Grain1Low = np.where(a.poslist[:, 2] < GB2pos)[0]
        Grain2 = np.where( (GB2pos <= a.poslist[:, 2]) & (a.poslist[:, 2] <= GB1pos) )[0]

        # generate all the structure files for RGS calculations
        # named as RGS_XXX.vasp. XXX represents different displacement.
        if isinstance(disBetwGrains, (list, np.ndarray) ):
            for dis in disBetwGrains:
                a = Latt.read_from_poscar()
                newcell = a.cell.copy()
                newcell.c[2] += dis*2
                a.set_cell(newcell, scaleatoms=False)
                a.set_direct(False)

                # Here we used the most rude way to apply the shift, pls be careful with the move method
                # move Grain1Up to the top of slab
                a.atomlist.move(Grain1Up, [0, 0, dis*2])
                # move the Grain2 with disBetwGrains
                a.atomlist.move(Grain2, [0, 0, dis])
                a.write_to_poscar("RGS_%5.3f.vasp" % dis)
        else:
            # create a vacuum layer with double the required distance first
            newcell = a.cell.copy()
            newcell.c[2] += disBetwGrains*2
            a.set_cell(newcell, scaleatoms=False)
            a.set_direct(False)

            # Here we used the most rude way to apply the shift, pls be careful with the move method
            # move Grain1Up to the top of slab
            a.atomlist.move(Grain1Up, [0, 0, disBetwGrains*2])
            # move the Grain2 with disBetwGrains
            a.atomlist.move(Grain2, [0, 0, disBetwGrains])
            a.write_to_poscar("RGS_%5.3f.vasp" % disBetwGrains)
        
        # using the structure files to generate the caculation list
        cal_list = []
        for i in os.listdir():
            if 'RGS' in i and i.split('.')[-1] == 'vasp':
                cal_list.append(i)

        process_name = 'RGS'

        for i in cal_list:
            serial_number = None
            cwd = os.getcwd()
            if len(i.split('_')) in [2, 3] :
                if not os.path.exists('./%s' % process_name):
                    create_path('./%s' % process_name, back=False)
                serial_number = i.split('RGS_')[-1].split('.vasp')[0]
                if os.path.exists('./%s/%s' % (process_name, serial_number)):
                    if skip_calculated:
                        continue
                create_path('./%s/%s' % (process_name, serial_number), back=True)
                os.system('cp %s %s/%s/POSCAR' % (i, process_name, serial_number))
                os.system('cp ../INCAR %s/%s' % (process_name, serial_number))
                os.system('cp ../KPOINTS %s/%s' % (process_name, serial_number))
                os.system('cp ../POTCAR %s/%s' % (process_name, serial_number))
                os.chdir('./%s/%s' % (process_name, serial_number))
                print('I am in %s'%(os.getcwd()))
                # this is RGS submit
                os.system('touch vasp_630.slurm')
                # pls change the command to submit the script
                os.chdir(cwd)


        
#---------------------------------------------------------------------#
#Main process
#---------------------------------------------------------------------#
root = os.getcwd()
for ele in element_lists:
    elePath = str(int(paraDict[ele]["AtomicNumber"])) + "-" + ele
    if not os.path.isdir(elePath):
        os.mkdir(elePath)
    os.chdir(elePath)
    eleRoot = os.getcwd()
    for calc in calc_lists:
        if calc == 6 or calc == 7:
            tensile(stage=calc)
        elif calc == 8 or calc == 9:
            RGS(stage=calc, ele=ele, disBetwGrains=RGSdis)
        else:
            genInput(ele=ele, stage=calc)
            if calc == 2 or calc == 5:
                incar = Incar.from_file("INCAR")
                incar.update({"NSW": 800})
                incar.write_file("INCAR")
                callVaspkit(if_surf=True)
            else:
                callVaspkit()
        os.chdir(eleRoot)
    os.chdir(root)
