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
script_file_name : str = "paper_figure_01.py"

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()
# Convert start time to real-world time
real_world_start_time : str = time.strftime("%y-%m-%d_%X_%Z").replace("/", "-").replace(":", "-")

# Set the figure number
figure_number : int = 1
# Set the number of tasks
task_id : int = 1
# Set the method for generating gates
method : str = 'interpolated_INC'
# Set the collapse operator
collapse_operator : str = 'z'
# Set dimensions
dims : list[int] = [4]
# Set the range of gamma values for the collapse operator
gamma_min : float = 1e-3 
gamma_max : float = 0.5
gamma_num : int = 1 * n_CPUs
gammas_array : np.ndarray = np.linspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
gammas : list[float] = gammas_ndarray.tolist()
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set the number of gates
n_gates : int = 4
# Set the time parameter
times : float = 1.0
# Set the superoperator status
super : bool = True
# Set the maximum number of iterations for the nested commutator sum
s_max : int = 60
# Set the error threshold for the nested commutator sum
error_threshold : float = 1e-8
# Set parallelization parameters
parallel_verbosity : int = 11
n_jobs_gates : int = min(n_CPUs, n_gates)
batch_size_gates : int = int(n_gates / n_jobs_gates)
n_jobs_gammas : int = int(n_CPUs)
batch_size_gammas : int = int(n_gammas / n_jobs_gammas)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.01) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
AGIs_d : list = []
AGIs_d_first_order : list = []
AGIs_d_second_order : list = []
super_gates_d : list = []
hamiltonians_d : list = []

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
print(f"COLLAPSE OPERATOR : {collapse_operator}")
print(f"NUMBER OF GAMMAS : {n_gammas}")
print(f"NUMBER OF GATES : {n_gates}")
print(f"GAMMAS : {gammas}")
print(f"GATE TIMES : {times}")
print('\n')
print(f'SUPEROPERATOR : {super}')
print(f'MAX ITERATIONS OVER S : {s_max}')
print(f'ERROR THRESHOLD : {error_threshold}')
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS FOR INITIAL STATES : {n_jobs_gates}")
print(f"BATCH SIZE PER STATE JOB : {batch_size_gates}")
print(f"NUMBER OF PARALLEL JOBS FOR FIDELITIES : {n_jobs_gammas}")
print(f"BATCH SIZE PER FIDELITY JOB : {batch_size_gammas}")
print(f"OPTIONS FOR MESOLVE : {options_mesolve}")
print('\n')

# Print the parameters
print(f'FIGURE_NUMBER: {figure_number:02}')
print('\n')

print(f"Gate Generation Method: {method}")
print (f"Dimensions: {dims}")
print(f"Gammas: {gammas}")
print(f"Times: {times}")
print(f"Super: {super}")
print(f"Maximum number of iterations: {s_max}")
print(f"Error threshold: {error_threshold}")
print('\n')

# INITIALIZATION TIME
initialization_time : float= time.time()
print(f"Initialization time: {initialization_time - start_time}")

print('\n')

# Loop over dimensions
for d in dims:

    # LOOP START TIME
    loop_start_time : float = time.time()  
    
    print(f"Dimension: {d}")

    # Initialize a qudit of dimension d
    qudit = Qudit(d = d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j(collapse_operator) # type: ignore 

    # Set the range of eta values for the gates
    etas_list : list[float] = [0.0, 3/4 - 1/(2 * d), 1.0]
    
    # Pre-allocate space for local data
    AGF : np.ndarray = np.empty(n_gammas, dtype = np.float64)
    AGI_first_order : np.ndarray = np.empty(gamma_num, dtype = np.float64)    
    AGI_second_order : np.ndarray = np.empty(n_gammas, dtype = np.float64) 
    comm_sum_second_order : np.ndarray = np.empty(n_gammas, dtype = np.float64)
    AGIs : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    AGIs_first_order : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    AGIs_second_order : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)    

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")

    # Generate gates
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_jobs_gates, batch_size = batch_size_gates, is_super = True, etas = etas_list)
    super_gates_2, hamiltonians_2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = True)

    super_gates = super_gates + super_gates_2
    hamiltonians = hamiltonians + hamiltonians_2

    super_gates_d.append(super_gates)
    hamiltonians_d.append(hamiltonians)

    # GATE GENERATION TIME
    gate_generation_time : float = time.time()
    print(f"Gate generation time: {gate_generation_time - initialization_time}")    

    # Calculate the first-order AGI correction term    
    AGI_first_order = -(gammas_array * times) * first_order_AGI(d, L)    

    # Loop over number of gates
    for idx, gate in enumerate(super_gates):

        # GATE FIDELITY START TIME
        gate_fidelity_start_time : float = time.time()

        # Parallelize the gammas loop
        AGF = Parallel(n_jobs = n_jobs_gammas, batch_size = batch_size_gammas, verbose = parallel_verbosity)(delayed(compute_fidelity)(hamiltonians[idx], gate, L, times, d, g, options_mesolve) for g in gammas) # type : ignore

        # Store the simulated AGI values
        AGIs[idx] = 1 - np.array(AGF)

        # Store the first-order AGI correction term
        AGIs_first_order[idx] = AGI_first_order

        # Calculate the second-order AGI correction terms
        comm_sum_second_order, s_finals = second_order_AGI(hamiltonians[idx], L, d, times, s_max, error_threshold = error_threshold, super = super)
        AGI_second_order = (gammas_array * times)**2 * comm_sum_second_order         
        AGIs_second_order[idx] = -AGI_second_order + AGI_first_order   

        # GATE FIDELITY TIME
        gate_fidelity_time : float = time.time()
        print(f"Gate: {idx + 1:03} / {n_gates:03} completed in: {gate_fidelity_time - gate_fidelity_start_time}") 

    # Store the AGI data per dimension
    AGIs_d.append(AGIs)
    AGIs_d_first_order.append(AGIs_first_order)
    AGIs_d_second_order.append(AGIs_second_order)

    # FIDELITY TIME
    fidelity_time : float = time.time()
    print(f"Fidelity time: {fidelity_time - gate_generation_time}")

# TOTAL LOOP TIME
total_loop_time : float = time.time()
print(f"Total loop time: {total_loop_time - start_time}")

# Save the data
if save_figure_data:
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gates = n_gates, s_max = s_max, error_threshold = error_threshold, AGIs_d = AGIs_d, AGIs_d_first_order = AGIs_d_first_order, AGIs_d_second_order = AGIs_d_second_order, hamiltonians_d = hamiltonians_d)

# SAVE DATA TIME
save_data_time : float = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time : float = time.time()
print(f"Total time: {total_time - start_time}")