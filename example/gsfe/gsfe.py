from matmec.core.latt import Latt
import os
import numpy as np
from dpdispatcher import Machine, Resources, Task, Submission
import glob
from monty.serialization import loadfn, dumpfn
import json
import shutil


def gen_task(task_json_txt, task_path):
    task_json=json.loads(task_json_txt)
    task_json_path = os.path.join(task_path, "task.json")
    with open(task_json_path,'w') as f:
        json.dump(task_json,f,indent=4)
    task_new = Task.load_from_json(task_json_path)

    return task_new


machine_json_txt='''
{
        "batch_type": "DpCloudServer",
        "context_type": "DpCloudServerContext",
        "local_root" : "./",
        "remote_profile":{
          "email": "yixuanhu97@sjtu.edu.cn",
          "password": "19971012tt@",
          "program_id": 10187,
            "input_data":{
              "api_version":2,
              "job_type": "indicate",
              "log_file": "*/vasp.out",
              "grouped":true,
              "job_name": "VASP",
              "disk_size": 100,
              "scass_type":"c32_m128_cpu",
              "platform": "ali",
              "image_name":"LBG_VASP_5.4.4_v1.1",
              "on_demand":0
        }
    }
}
'''
resources_json_txt='''
{
        "number_node": 126498,
        "cpu_per_node": 32,
        "gpu_per_node": 0,
        "queue_name": "LBG",
        "group_size": 1,
        "source_list": ["/opt/intel/oneapi/setvars.sh"]
}
'''

rootdir = os.getcwd()
# pls give the path for INCAR, POTCAR and KPOINTS
incar_path = "./INCAR"
potcar_path = "./POTCAR"
kpoint_path = "./KPOINTS"

incar = os.path.abspath(incar_path)
potcar = os.path.abspath(potcar_path)
kpoint = os.path.abspath(kpoint_path)

if not os.path.exists('./basal'):
    os.mkdir('./basal')

os.system('cp ./POSCAR ./basal/')
os.chdir('./basal')

a = Latt()
a.read_from_poscar()
a.set_fix_TopBottom()
a.write_to_poscar("POSCAR_00.vasp")

# pls make sure you know which direction you're shifting the cell
moveDirec = (a.cell.lattvec[0]/np.linalg.norm(a.cell.lattvec[0]) - a.cell.lattvec[1]/np.linalg.norm(a.cell.lattvec[1]))

# pls give a proper distance to move and proper lowLimit
step = 0.8660260862854056/20
lowerLimit = 0.360

# for i in [1, 2, 3, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 7, 8, 9, 10]:
for i in range(1, 21):
    a.read_from_poscar()
    a.move_Part(moveDirec, i*step, lowerLimit)
    a.set_fix_TopBottom()
    a.write_to_poscar('POSCAR_%02d.vasp' %i )

process_name = "basal"
poscar_list = glob.glob("POSCAR_*.vasp")

work_list = []
# make working paths, copy the required POSCAR, INCAR, POTCAR, KPOINTS
# generating work_list, tells what folders contain work files
for poscar in poscar_list:
    ind = poscar.split(".vasp")[0].split("POSCAR_")[-1]
    dir_name = os.path.abspath(process_name + "_" + ind)
    os.makedirs(dir_name, exist_ok=True)
    # os.system("cp %s %s" %(poscar, os.path.join(dir_name, "POSCAR")))
    # os.system("cp %s %s" %(incar, os.path.join(dir_name, "INCAR")))
    # os.system("cp %s %s" %(potcar, os.path.join(dir_name, "POTCAR")))
    # os.system("cp %s %s" %(kpoint, os.path.join(dir_name, "KPOINTS")))
    shutil.copy(poscar, os.path.join(dir_name, "POSCAR"))
    shutil.copy(incar, os.path.join(dir_name, "INCAR"))
    shutil.copy(kpoint, os.path.join(dir_name, "KPOINTS"))
    shutil.copy(potcar, os.path.join(dir_name, "POTCAR"))
    work_list.append(dir_name)
    del dir_name

# using the work_list
# generate the task list for dpdispatcher
task_list = []
for task in work_list:
    task_work_path = task
    task_json_txt = '''
    "command": "source /opt/intel/oneapi/setvars.sh --force && mpirun -np 16 vasp_std",
    "task_work_path": "{task_work_path}",
    "forward_files": [],
    "backward_files": [],
    "outlog": "vasp.log",
    "errlog": "vasp.err"
    '''.format(task_work_path=task_work_path)
    task_json_txt = "{"+task_json_txt+"}"
    task_new = gen_task(task_json_txt, task_work_path)
    task_list.append(task_new)


# submit
machine_json = json.loads(machine_json_txt)
resources_json = json.loads(resources_json_txt)
machine = Machine.load_from_dict(machine_json)
resources = Resources.load_from_dict(resources_json)
submission = Submission(
    work_base = "./",
    machine = machine,
    resources = resources,
    task_list = task_list,
    forward_common_files = [],
    backward_common_files = []
)

# submission.run_submission()

os.system('rm *sub')
os.system('rm -r backup')
os.system('rm *fail')
os.system('rm *finished')