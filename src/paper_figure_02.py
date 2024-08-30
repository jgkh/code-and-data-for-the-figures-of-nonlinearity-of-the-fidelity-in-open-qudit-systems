# ----------------------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------------------

import paper_imports
import paper_methods

from paper_imports import *
from paper_methods import *

importlib.reload(paper_imports)
importlib.reload(paper_methods)

# ----------------------------------------------------------------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------------------------------------------------------------

# Save data to file
save_figure_data : bool = True

# ----------------------------------------------------------------------------------------------------------------------------------
# HPC SETTINGS
# ----------------------------------------------------------------------------------------------------------------------------------

try:
    
    task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    
except:
    
    task_count = 1
try:
    
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])   
    
except:
    
    task_id = 1

try:
    
    cpus_on_node = int(os.environ["SLURM_CPUS_ON_NODE"])
    
except:
    
    cpus_on_node = 'frontend'

if isinstance(cpus_on_node, int):
    n_CPUs : int = cpus_on_node
else:
    n_CPUs : int = int(os.cpu_count()) # type:ignore

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIME
start_time = time.time()

# Set the figure number
figure_number = 12
# Set the method for generating gates
method = 'haar'
# Set dimension for the quantum system (in this case, a qubit)
dims = [2, 4]
# Set the range of noise strengths for the collapse operator
gamma_min = 1e-6
gamma_max = 1e+3
gamma_num = 28
gammas = np.geomspace(gamma_min, gamma_max, gamma_num).tolist()
# Set the times for the propagator
times = 1.0
# Set parallelization parameters
n_gates = n_CPUs
n_jobs = min(n_CPUs, n_gates)
batch_size = int(n_gates / n_jobs) 
# Set options for the mesolve function (solver options for differential equations)
options_mesolve = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
AGIs_d = []
super_gates_d = []
hamiltonians_d = []

# Print the parameters
print(f'FIGURE_NUMBER: {figure_number:02}')
print('\n')

print(f"Gate Generation Method: {method}")
print (f"Dimensions: {dims}")
print(f"Gammas: {gammas}")
print(f"Times: {times}")
print('\n')

print(f"Number of CPUs: {n_CPUs}")
print(f"Number of gates: {n_gates}")
print(f"Number of jobs: {n_jobs}")
print(f"Batch size: {batch_size}")
print(f"Task number: {task_id:03} / {task_count:03}")
print('\n')

# INITIALIZATION TIME
initialization_time = time.time()
print(f"Initialization time: {initialization_time - start_time}")

print('\n')

# Loop over the dimensions in dims
for d in dims:    

    # LOOP START TIME
    loop_start_time = time.time()    

    print(f"Dimension: {d}")   

    # Initialize a qudit of dimension d
    qudit = Qudit(d=d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j("z") # type: ignore

    # Pre-allocate space for AGIs
    AGIs = np.empty((n_gates, len(gammas)), dtype=np.float64)
    # Pre-allocate space for AGF
    AGF = np.empty((n_gates, len(gammas)), dtype=np.float64)

    # LOOP INITIALIZATION TIME
    loop_initialization_time = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")    

    # Generate a list of random unitary gates and Hamiltonians
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs, batch_size = batch_size, is_super = True)

    super_gates_d.append(super_gates)
    hamiltonians_d.append(hamiltonians)

    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - loop_initialization_time}")

    # Loop over number of gammas
    for g_idx, g in enumerate(gammas):
        # Parallelize the gates loop
        AGF[:, g_idx] = Parallel(n_jobs = n_jobs, batch_size = batch_size)(delayed(compute_fidelity)(hamiltonians[idx], gate, L, times, d, g, options_mesolve) for idx, gate in enumerate(super_gates)) # type : ignore        
        AGIs[:, g_idx] = 1 - np.array(AGF[:, g_idx])

    # Store the AGI data per dimension
    AGIs_d.append(AGIs)

    # FIDELITY TIME
    fidelity_time = time.time()
    print(f"Fidelity time: {fidelity_time - gate_generation_time}")

    # SINGLE LOOP TIME
    loop_time = time.time()
    print(f"Single loop time: {loop_time - loop_start_time}")

    print('\n')

# TOTAL LOOP TIME
total_loop_time = time.time()
print(f"Total loop time: {total_loop_time - start_time}")

# Save the data
if save_figure_data:
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gates = n_gates, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d, super_gates_d = super_gates_d)

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------