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
script_file_name : str = os.path.basename(__file__)

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

# Set the figure number
figure_number : int = 18
# Find existing data files before overwriting
files_found = [name for name in os.listdir('../dat/paper_2') if name.startswith(f'Figure_{figure_number:02}')]
print(f'Existing Data Files Found: {files_found}')
n_files = len(files_found)
# Round n_files down to the nearest multiple of 10, since we work in batches of 10 and tasks may run asynchronously
#n_files = n_files - n_files % 10
n_files = 30 # hard-code to known value to prevent bugs between multiple concurrent tasks

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()


# Set the method for generating gates
method : str = 'haar'
# Set max dimension for the qudits
d_max : int = 64
# Set dimensions for the qudits
dims : list[int] = np.arange(2, d_max + 1, 1).tolist()
# Set number of dimensions
n_dims : int = len(dims)
# Set collapse operator
collapse_operator : str = "z" # ['x', 'y','z']
# Set alias for tasks
n_tasks : int = task_count
# Set the number of gates
n_gates : int = 2 * n_CPUs
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Set maximum value of s for the integration, and error threshold
s_max : int = 60
# Set error threshold
error_threshold : float = 1e-8
# Set parallelization parameters
parallel_verbosity : int = 11
n_jobs : int = min(n_CPUs, n_gates)
batch_size : int = int(n_gates / n_jobs)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
result = []
AGIs_d : list = []
super_gates_d : list = []
hamiltonians_d : list = []
s_finals_d : np.ndarray = np.zeros((n_dims, n_gates), dtype = int)

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
print(f"NUMBER OF DIMENSIONS : {n_dims}")
print('\n')
print (f"QUDIT DIMENSIONS : {dims}")
print(f"TIMES : {times}")
print('\n')
print(f"MAXIMUM VALUE OF s : {s_max}")
print(f"ERROR THRESHOLD : {error_threshold}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS : {n_jobs}")
print(f"BATCH SIZE PER JOB : {batch_size}")
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
    # Assign pure dephasing operator to L
    L : qt.Qobj = j(collapse_operator) # type:ignore

    # Initialise the etas list for the interpolated_INC method
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

    # Generate list of unitary matrices and hamiltonians
    if method == 'haar':
        super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs, batch_size = batch_size, is_super = False)
    else:        
        super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_jobs, batch_size = batch_size, is_super = False, etas = etas_list)
        super_gates_2, hamiltonians_2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = False)
        super_gates = super_gates + super_gates_2
        hamiltonians = hamiltonians + hamiltonians_2

    super_gates_d.append(super_gates)
    hamiltonians_d.append(hamiltonians)
    
    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - loop_initialization_time}")

    #Parallelize second_order_AGI over gates
    result = Parallel(n_jobs = n_jobs, batch_size = batch_size, return_as = 'list', verbose = parallel_verbosity)(delayed(second_order_AGI)(hamiltonians_d[d_idx][gate_idx], L, d, times, s_max, error_threshold = error_threshold, super = True) for gate_idx in range(n_gates)) # type : ignore

    s_finals_d[d_idx, :] = np.array([result[gate_idx][1] for gate_idx in range(n_gates)]) # type: ignore

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

# Update task_id to account for the tasks that have already been run
task_id += n_files

# Save the data
if save_figure_data:
    save_data(figure_number, task_id, dims = dims, n_gates = n_gates, times = times, hamiltonians_d = hamiltonians_d, s_finals_d = s_finals_d)

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------
