import os
import subprocess as sp
import time
import glob
import shutil

def write_dp_in(dp_in, dp):
    with open(dp_in, 'r') as f:
        line = f.readline()
        s = ''
        while line:
            if "pair_style" in line and line[0] != '#':
                s += 'pair_style      deepmd %s\n' % dp
            else:
                s += line
            line = f.readline()
    with open('in.tmp', 'w') as f:
        f.write(s)

def write_eam_in(eam_in, eam, element):
    with open(eam_in, 'r') as f:
        line = f.readline()
        s = ''
        while line:
            if "pair_style" in line and line[0] != '#':
                s += 'pair_style      eam/fs\n'
                s += "pair_coeff      * * %s %s\n" % (eam, element)
                f.readline()
            else:
                s += line
            line = f.readline()
    with open('in.tmp', 'w') as f:
        f.write(s)

dp_list = ["Ti5_small_i.pb" , "Ti5_small_i2.pb", "Ti5_big_i.pb", "Ti5_big_i2.pb"] #, "Ti5_attn_i.pb", "Ti5_attn.pb"]
dp_path_list = [ os.path.abspath(dp) for dp in dp_list ]

eam_list = []# ['Ti.eam.fs']
eam_path_list = [ os.path.abspath(eam) for eam in eam_list ]

element = 'Ti'

work_list = glob.glob("*/dowork")
work_list = [ work.split('dowork')[0] for work in work_list ]
root = os.getcwd()

for work in work_list:
    os.chdir(work)
    for i in os.listdir():
        if i[:3] == 'in.' and i[-3:] == '_dp':
            dp_in = i
        elif i[:3] == 'in.' and i[-4:] == '_eam':
            eam_in = i
    for i, dp in enumerate(dp_path_list):
        dp = os.path.relpath(dp)
        write_dp_in(dp_in, dp)
        proc = sp.Popen(["lmp", "-i", "in.tmp"], stdin=sp.PIPE)
        while proc.poll() == None:
            time.sleep(3)
        shutil.copy("gsfe",  "%s_gsfe" % dp_list[i])
        os.remove("gsfe")
        os.remove("in.tmp")

    for i, eam in enumerate(eam_path_list):
        eam = os.path.relpath(eam)
        write_eam_in(eam_in, eam, element)
        proc = sp.Popen(["lmp", "-i", "in.tmp"], stdin=sp.PIPE)
        while proc.poll() == None:
            time.sleep(3)
        try:
            shutil.copy("gsfe",  "%s_gsfe" % eam_list[i])
        except:
            raise RuntimeError("Check the %s folder" % os.path.abspath(os.getcwd()).split('/')[-1])
        os.remove("gsfe")
        os.remove("in.tmp")
    os.chdir(root)