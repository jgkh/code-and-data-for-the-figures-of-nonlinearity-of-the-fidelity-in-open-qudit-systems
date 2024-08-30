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
# SLURM SETTINGS
# ----------------------------------------------------------------------------------------------------------------------------------

# Get script file name
script_file_name : str = os.path.basename(__file__)

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

# Hard-code task_id for alternative data files
n_tasks = 10
task_id += n_tasks

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()

# Set the figure number
figure_number : int = 19
# Set the method for generating gates
method : str = 'haar' # interpolated_INC, 
# Set max dimension for the qudits
d_max : int = 48
# Set dimensions for the qudits
dims : list[int] = np.arange(2, d_max + 1, 1).tolist()
# Set number of dimensions
n_dims : int = len(dims)
# Set collapse operator
collapse_operator : str = "z" # ['x', 'y','z']
# Set alias for tasks
n_tasks : int = task_count
# Set the number of gates
n_gates : int = 10
# Set times for the main figure
t_min : float = 0.0
t_max : float = 5.0
n_times : int = 1 * n_CPUs
times : list[float] = np.linspace(t_min, t_max, n_times).tolist()
# Set times for the inset figure
t_min_inset : float = 0.0
t_max_inset : float = 0.2
n_times_inset : int = n_CPUs
times_inset : list[float] = np.linspace(t_min_inset, t_max_inset, n_times_inset).tolist()
# Set maximum value of s for the integration, and error threshold
s_max : int = 300
# Set error threshold
error_threshold : float = 1e-8
# Set parallelization parameters
parallel_verbosity : int = 11
# Gate generation parallelization parameters
n_jobs_gate_generation : int = min(n_CPUs, n_gates)
batch_size_gate_generation : int = int(n_gates / n_jobs_gate_generation)
# Time parallelization parameters
n_jobs_times : int = min(n_CPUs, n_times)
batch_size_times : int = int(n_times / n_jobs_times)
# Insets time parallelization parameters
n_jobs_times_inset : int = min(n_CPUs, n_times_inset)
batch_size_times_inset : int = int(n_times_inset / n_jobs_times_inset)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
result = []
result_inset = []
hamiltonians_d : list = []
s_finals_d : np.ndarray = np.zeros((n_dims, n_gates, n_times), dtype = int)
s_finals_d_inset : np.ndarray = np.zeros((n_dims, n_gates, n_times_inset), dtype = int)
AGI2_finals_d : np.ndarray = np.zeros((n_dims, n_gates, n_times), dtype = float)

# Print the config parameters
print('PRINTING CONFIG PARAMETERS:')
print('---------------------------')
print('\n')
print(f"CURRENT TASK : {task_id:03} / {task_count:03}")
print(f'MODIFIED FIGURE NUMBER : {figure_number}')
print(f'SAVING FIGURE DATA : {save_figure_data}')
print('\n')
print(f"GATE GENERATION METHOD : {method}")
print(f"COLLAPSE OPERATOR : {collapse_operator}")
print('\n')
print(f"NUMBER OF GATES : {n_gates}")
print(f"NUMBER OF TIMES FOR MAIN FIGURE : {n_times}")
print(f"NUMBER OF TIMES FOR INSET FIGURE : {n_times_inset}")
print(f"NUMBER OF DIMENSIONS : {n_dims}")
print('\n')
print (f"QUDIT DIMENSIONS : {dims}")
print(f"MAIN FIGURE TIMES : {times}")
print(f"INSET FIGURE TIMES : {times_inset}")
print('\n')
print(f"MAXIMUM VALUE OF s : {s_max}")
print(f"ERROR THRESHOLD : {error_threshold}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS FOR GATE GENERATION : {n_jobs_gate_generation}")
print(f"BATCH SIZE PER WORKER FOR GATE GENERATION : {batch_size_gate_generation}")
print(f"NUMBER OF PARALLEL JOBS FOR MAIN FIGURE TIMES : {n_jobs_times}")
print(f"BATCH SIZE PER WORKER FOR MAIN FIGURE TIMES : {batch_size_times}")
print(f"NUMBER OF PARALLEL JOBS FOR INSET FIGURE TIMES : {n_jobs_times_inset}")
print(f"BATCH SIZE PER WORKER FOR INSET FIGURE TIMES : {batch_size_times_inset}")
print('\n')
print(f"OPTIONS FOR MESOLVE : {options_mesolve}")
print('\n')

# INITIALIZATION TIME
initialization_time : float = time.time()
print(f"Initialization time: {initialization_time - start_time}")
print('\n')

# Loop over the dimensions 
for d_idx, d in enumerate(dims):  

    # LOOP START TIME
    loop_start_time : float = time.time()

    print(f"DIMENSION : {d:02} / {d_max:02}")

    # Initialize a Qudit of dimension d
    qudit = Qudit(d=d)
    # Get collapse operators related to the Qudit
    j = qudit.get_j
    # Assign collapse operator to L
    L : qt.Qobj = j(collapse_operator) # type:ignore

    # Initialise the etas list for the interpolated INC gates
    etas_list = []
    if method == 'interpolated_INC':
        etas_list = (3 * d - 2) / (4 * d)
        etas_list = np.linspace(0, etas_list, n_gates - 3)
        etas_list = np.append(etas_list, (etas_list[-1] + etas_list[-2]) / 2)
        etas_list = np.append(etas_list, 1)
        etas_list = np.sort(etas_list).tolist()
        print(f"ETAS LIST : {etas_list}")

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")  

    # Generate a list of gates and hamiltonians for the current dimension and method
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs_gate_generation, batch_size = batch_size_gate_generation, is_super = False)
    # Generate 1 gate and hamiltonian for the superposition (Hadamard/ Crestenson/ QFT) gate
    # super_gates_2, hamiltonians_2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = False)

    # Concatenate the hamiltonians
    # hamiltonians = hamiltonians + hamiltonians_2
    # Append the hamiltonians to the list per dimension
    hamiltonians_d.append(hamiltonians)
    
    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - loop_initialization_time}")

    # Loop over number of gates
    for gate_idx in range(n_gates):
                
        print(f"GATE : {gate_idx + 1:03} / {n_gates:03}")

        # Parallelize second_order_AGI over dimensions for main figure times
        result = Parallel(n_jobs = n_jobs_times, batch_size = batch_size_times, return_as = 'list', verbose = parallel_verbosity)(delayed(second_order_AGI)(hamiltonians_d[d_idx][gate_idx], L, d, t, s_max, error_threshold = error_threshold, super = True) for t in times) # type : ignore

        # Store the final s values for the current gate and dimension
        s_finals_d[d_idx, gate_idx, :] = np.array([result[t_idx][1] for t_idx, t in enumerate(times)]) # type : ignore
        AGI2_finals_d[d_idx, gate_idx, :] = np.array([result[t_idx][0] for t_idx, t in enumerate(times)]) # type : ignore

        # Parallelize second_order_AGI over dimensions for inset figure times
        # result_inset = Parallel(n_jobs = n_jobs_times_inset, batch_size = batch_size_times_inset, return_as = 'list', verbose = parallel_verbosity)(delayed(second_order_AGI)(hamiltonians_d[d_idx][gate_idx], L, d, t, s_max, error_threshold = error_threshold, super = True) for t in times_inset) # type : ignore

        # Store the final s values for the current gate and dimension for the inset figure
        # s_finals_d_inset[d_idx, gate_idx, :] = np.array([result_inset[t_idx][1] for t_idx, t in enumerate(times_inset)])

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
    save_data(figure_number, task_id, dims = dims, n_gates = n_gates, hamiltonians_d = hamiltonians_d, times = times, s_finals_d = s_finals_d, AGI2_finals_d = AGI2_finals_d)#, times_inset = times_inset, s_finals_d_inset = s_finals_d_inset)

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------
