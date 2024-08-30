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
script_file_name : str = "paper_figure_13.py"

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

n_tasks = 0
task_id += n_tasks

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()
# Convert start time to real-world time
real_world_start_time : str = time.strftime("%y-%m-%d_%X_%Z").replace("/", "-").replace(":", "-")

# Set the figure number
figure_number : int = 83
# Set the method for generating gates
method : str = 'haar'
# Set dimensions for the qudits
dims : list[int] = [8]
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-2
gamma_num : int = 100
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set list of max gamma values
gamma_max_list : list[float] = [1e3]*1
# Set collapse operator
collapse_operator : str = "z" # ['x', 'y','z']
# Set number of random quantum gates
n_gates : int = 10000
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Error threshold for root finding the zeros of the gate infidelity
y_threshold : float = 1e-3
# Set parallelization parameters
parallel_verbosity : int = 11
n_jobs_gates : int = min(n_CPUs, n_gates)
batch_size_gates : int = int(n_gates / n_jobs_gates)
n_jobs_fidelities : int = min(n_CPUs, n_gates)
batch_size_fidelities : int = int(n_gates / n_jobs_fidelities)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance
# Set list of options.mesolve nsteps values
nsteps_list : list[float] = [1e6]*1

# Print the config parameters
print(f"STARTING TIMESTAMP: {real_world_start_time}")
print('\n')
print('PRINTING CONFIG PARAMETERS:')
print('---------------------------')
print('\n')
print(f"CURRENT TASK : {task_id:03} / {task_count:03}")
print(f'MODIFIED FIGURE NUMBER : {figure_number}')
print(f'SAVING FIGURE DATA : {save_figure_data}')
print('\n')
print(f"GATE GENERATION METHOD : {method}")
print (f"QUDIT DIMENSIONS : {dims}")
# print(f"GAMMAS : {gammas}")
print(f"COLLAPSE OPERATOR : {collapse_operator}")
print(f"NUMBER OF GATES : {n_gates}")
print(f"GATE TIMES : {times}")
print(f"ERROR THRESHOLD : {y_threshold}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS : {n_jobs_gates}")
print(f"BATCH SIZE PER JOB : {batch_size_gates}")
print(f"NUMBER OF PARALLEL JOBS : {n_jobs_fidelities}")
print(f"BATCH SIZE PER JOB : {batch_size_fidelities}")
print(f"OPTIONS FOR MESOLVE : {options_mesolve}")
print('\n')

# Pre-allocate space for data
AGIs_d : list = []
hamiltonians_d : list = []

# INITIALIZATION TIME
initialization_time : float = time.time()
print(f"INITIALISATION TIME: {initialization_time - start_time:.4f}")

print('\n')

# Loop over the dimensions
for d_idx, d in enumerate(dims):

    # LOOP START TIME
    loop_start_time : float = time.time()  
    
    print(f"DIMENSION : {d:02} / {dims}")

    # Increase nsteps for larger dimensions
    # This is to ensure that the solver can handle the increased dimensionality
    # And avoid ODE integration errors of the zvode solver
    options_mesolve.nsteps = float(nsteps_list[d_idx]) # type: ignore

    # Decrease the gamma_max for larger dimensions
    # This is to ensure that the solver can handle the increased dimensionality
    # And avoid excessive computation times for large gammas and dimensions
    gamma_max = gamma_max_list[d_idx]
    gammas_array : np.ndarray = np.geomspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
    gammas : list[float] = gammas_array.tolist()

    # Initialize a qudit of dimension d
    qudit = Qudit(d = d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j(collapse_operator) # type: ignore

    # Pre-allocate space for local data for the current dimension
    AGIs : np.ndarray = np.zeros((n_gates, n_gammas), dtype = np.float64) 

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time:.4f}")   

    # Generate a list of random unitary matrices
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs_gates, batch_size = batch_size_gates, is_super = True)
    # Store the super gates and Hamiltonians for the current dimension
    hamiltonians_d.append(hamiltonians)    

    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - loop_initialization_time:.4f}")

    # Prepare parallel computation tasks
    tasks = [(gamma_idx, gate_idx, hamiltonians[gate_idx], gate, L, times, d, gamma, options_mesolve) for gamma_idx, gamma in enumerate(gammas) for gate_idx, gate in enumerate(super_gates)]
    # Execute tasks in parallel
    fidelity_results = Parallel(n_jobs = n_CPUs, batch_size='auto', verbose = parallel_verbosity)(delayed(compute_fidelity_wrapper)(*task) for task in tasks)    
    # Collect results
    for gamma_idx, gate_idx, fidelities in fidelity_results:
        AGIs[gate_idx, gamma_idx] = 1 - fidelities
    AGIs_d.append(AGIs)

    # FIDELITY TIME
    fidelity_time = time.time()
    print(f"Fidelity time: {fidelity_time - gate_generation_time:.4f}")

    # Perform garbage collection to free up memory
    gc.collect()
    # Clear large variables to free up memory
    del qudit, super_gates, hamiltonians, AGIs, tasks, fidelity_results
    gc.collect()

    # SINGLE LOOP TIME
    loop_time = time.time()
    print(f"Single loop time: {loop_time - loop_start_time:.4f} seconds\n")

    print('\n')

# TOTAL LOOP TIME
total_loop_time = time.time()
print(f"Total loop time: {total_loop_time - start_time:.4f}")

# Save the data
if save_figure_data:    
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gates = n_gates, times = times, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d)    
# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time:.4f}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time:.4f}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------