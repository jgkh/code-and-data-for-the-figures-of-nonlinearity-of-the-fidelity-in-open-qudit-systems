# ----------------------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------------------

import paper_imports
import paper_methods

from paper_imports import *
from paper_methods import *

importlib.reload(paper_imports)
importlib.reload(paper_methods)

import paper_imports
import paper_methods

from paper_imports import *
from paper_methods import *

# ----------------------------------------------------------------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------------------------------------------------------------

# Save data to file
save_figure_data : bool = True

# ----------------------------------------------------------------------------------------------------------------------------------
# HPC SETTINGS
# ----------------------------------------------------------------------------------------------------------------------------------

# Get script file name
script_file_name : str = "paper_figure_03.py"

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

n_tasks = 2
task_id += n_tasks

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()

# Set the figure number
figure_number : int = 3
# Set the method for generating gates
method : str = 'haar'
# Set dimensions for the qudits
dims : list[int] = [2]
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-2
gamma_max : float = 1e+4
gamma_num : int = 100
gammas_ndarray : np.ndarray = np.geomspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
gammas : list[float] = gammas_ndarray.tolist()
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set number of random quantum gates
n_gates : int = 10000
# Set parallelization parameters
n_jobs : int = n_CPUs
batch_size : int = int(n_gates / n_jobs)   
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
AGIs_d : list = []
super_gates_d : list = []
hamiltonians_d : list = []

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
initialization_time : float = time.time()
print(f"Initialization time: {initialization_time - start_time}")

print('\n')

# Loop over the dimensions
for d in dims:

    # LOOP START TIME
    loop_start_time : float = time.time()  
    
    print(f"Dimension: {d}")

    # Initialize a qudit of dimension d
    qudit = Qudit(d = d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j("z") # type: ignore    

    # Pre-allocate space for local data
    AGIs : np.ndarray = np.empty((n_gates, n_gammas), dtype=np.float64)
    AGF : np.ndarray = np.empty((n_gates, n_gammas), dtype=np.float64)

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")    

    # Generate a list of random unitary gates and Hamiltonians
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs, batch_size = batch_size, is_super = True) # method = 'circular' or 'random' or 'haar' or 'hermitian' or 'cirq_random_unitary' or 'cirq_random_special_unitary'

    super_gates_d.append(super_gates)
    hamiltonians_d.append(hamiltonians)

    # GATE GENERATION TIME
    gate_generation_time : float = time.time()
    print(f"Gate generation time: {gate_generation_time - initialization_time}")

    # Loop over number of gammas
    for idx_g, g in enumerate(gammas):

        # GAMMA FIDELITY START TIME
        gamma_fidelity_start_time : float = time.time()

        # Parallelize the gates loop
        AGF[:, idx_g] = Parallel(n_jobs = n_jobs, batch_size = batch_size)(delayed(compute_fidelity)(hamiltonians[idx], gate, L, times, d, g, options_mesolve) for idx, gate in enumerate(super_gates)) # type : ignore
        AGIs[:, idx_g] = 1 - AGF[:, idx_g]

        # GAMMA FIDELITY TIME
        gamma_fidelity_time : float = time.time()
        print(f"Gamma: {idx_g + 1:03} / {n_gammas:03} completed in: {gamma_fidelity_time - gamma_fidelity_start_time}") 

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
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gates = n_gates, times = times, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d)

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------