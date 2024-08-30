import os

import time, datetime

# ----------------------------------------------------------------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------------------------------------------------------------

SLURM = True

GRANT = True

TASK_COUNT = 1

SIMULTANEOUS_TASK_COUNT = 1

FIGURE_NUMBER = 3 # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

# ----------------------------------------------------------------------------------------------------------------------------------

if os.environ['USER'] == "jgkh":

    PYTHON_VERSION = "python3.9"

else:
    
    PYTHON_VERSION = "python"

NODE_COUNT = 1

CPU_COUNT = 24

LINE_BREAK = "\n\n"

SCRIPT_NAME = "paper_figure_{:02}.py".format(FIGURE_NUMBER)

FILE_TYPES = ['job', 'out', 'dat', 'cfg']

TIME_STAMP = time.strftime("%y-%m-%d_%X_%Z").replace("/", "-").replace(":", "-")

MAX_RUNTIME = "96:00:00" if GRANT else "24:00:00"

MEM_CONSTRAINT = "128G"

GRANT_NAME = "g2023a154c"

GRANT_ACCOUNT = "grant -A " + GRANT_NAME if GRANT else "public"

DIR_NAME = "../run/paper_figure_{:02}_{}".format(FIGURE_NUMBER, TIME_STAMP)

JOB_NAME = "{}/job/{:03}-TASKS-SUBMITTED-{}.sh".format(DIR_NAME, TASK_COUNT, TIME_STAMP)

EXEC_PYTHON = "{} ./{} {} {} {}".format(PYTHON_VERSION, SCRIPT_NAME, DIR_NAME, TIME_STAMP, FIGURE_NUMBER)

EXEC_SBATCH = "sbatch {}".format(JOB_NAME)


print(DIR_NAME)

os.mkdir(DIR_NAME)

for FILE in FILE_TYPES:
    
    os.mkdir("{}/{}".format(DIR_NAME, FILE))


with open(JOB_NAME, 'w') as io:
    
    io.writelines("#! /bin/bash")
    
    io.writelines(LINE_BREAK)        

    io.writelines("#SBATCH --job-name={}".format(JOB_NAME))
    
    io.writelines(LINE_BREAK)        

    io.writelines("#SBATCH --output={}/out/job-%A-%3a-of-{:03}-%j-%N-{}.out".format(DIR_NAME, TASK_COUNT, TIME_STAMP))

    io.writelines(LINE_BREAK)   

    io.writelines("#SBATCH --error={}/out/job-%A-%3a-of-{:03}-%j-%N-{}.err".format(DIR_NAME, TASK_COUNT, TIME_STAMP))

    io.writelines(LINE_BREAK)  

    io.writelines("#SBATCH --nodes={}".format(NODE_COUNT))

    io.writelines(LINE_BREAK)  

    io.writelines("#SBATCH --exclusive")

    io.writelines(LINE_BREAK)  

    io.writelines("#SBATCH --time={}".format(MAX_RUNTIME))

    io.writelines(LINE_BREAK)

    io.writelines("#SBATCH --mem={}".format(MEM_CONSTRAINT))

    io.writelines(LINE_BREAK)

    io.writelines("#SBATCH --cpus-per-task={}".format(CPU_COUNT))

    io.writelines(LINE_BREAK)

    io.writelines("#SBATCH -p {}".format(GRANT_ACCOUNT))

    io.writelines(LINE_BREAK)  

    io.writelines("#SBATCH --array=1-{}%{}".format(TASK_COUNT, SIMULTANEOUS_TASK_COUNT))

    io.writelines(LINE_BREAK)  

    io.writelines(EXEC_PYTHON)
    
        
os.system("module load python")

if (SLURM):

    os.system(EXEC_SBATCH)
    
else:
    
    os.system(EXEC_PYTHON)
