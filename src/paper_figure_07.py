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
script_file_name : str = "paper_figure_07.py"

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

# Find existing data files before overwriting
files_found = [name for name in os.listdir('../dat/paper_2') if name.startswith(f'Figure_{figure_number:02}')]
print(f'Existing Data Files Found: {files_found}')
n_files = len(files_found)
# Round n_files down to the nearest multiple of 10, since we work in batches of 10 and tasks may run asynchronously
n_files = n_files - n_files % 10

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()

# Convert start time to real-world time
real_world_start_time : str = time.strftime("%y-%m-%d_%X_%Z").replace("/", "-").replace(":", "-")

# Set the figure number
figure_number : int = 17
# Set the method for generating gates
method : str = 'haar' # 'haar' or 'interpolated_INC'
# Set dimensions for the qudits
dims : list[int] = [2, 4]
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-3
gamma_max : float = 1e-1
gamma_num : int = 50
gammas_ndarray : np.ndarray = np.linspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
gammas : list[float] = gammas_ndarray.tolist()
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set collapse operator
collapse_operator : str = "z" # ['x', 'y','z']
# Set number of random quantum gates
n_gates : int = 4 if method == 'interpolated_INC' else 10 * n_CPUs
# Set parallelization parameters
# Set parallelization parameters
parallel_verbosity : int = 11
n_jobs_gates : int = min(n_CPUs, n_gates)
batch_size_gates : int = int(n_gates / n_jobs_gates)
n_jobs_fidelities : int = min(n_CPUs, n_gammas)
batch_size_fidelities : int = int(n_gammas / n_jobs_fidelities)
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
AGIs_d_relative_error : list = []
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
print(f"COLLAPSE OPERATOR : {collapse_operator}")
print (f"QUDIT DIMENSIONS : {dims}")
print(f"NUMBER OF GATES : {n_gates}")
print(f"NUMBER OF GAMMAS : {n_gammas}")
print(f"GAMMAS : {gammas}")
print(f"GATE TIMES : {times}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS FOR GATES : {n_jobs_gates}")
print(f"BATCH SIZE PER GATE JOB : {batch_size_gates}")
print(f"NUMBER OF PARALLEL JOBS FOR FIDELITIES : {n_jobs_fidelities}")
print(f"BATCH SIZE PER FIDELITY JOB : {batch_size_fidelities}")
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
    
    print(f"Dimension: {d}")

    # Initialize a qudit of dimension d
    qudit = Qudit(d = d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j(collapse_operator) # type: ignore    

    # Pre-allocate space for local data for the current dimension
    AGF = np.empty(n_gammas, dtype = np.float64)
    AGI = np.empty(n_gammas, dtype = np.float64)
    AGIs = np.empty((n_gates, n_gammas), dtype = np.float64)    
    AGI_first_order = np.empty(len(gammas), dtype=np.float64)
    AGIs_1_relative_error = np.empty((n_gates, len(gammas)), dtype=np.float64)   
    super_gates = []
    hamiltonians = []

    # Calculate First Order AGI Correction
    AGI_first_order = (np.array(gammas) * times) * first_order_AGI(d, L)

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")  
    
    # Generate interpolated gates
    if method == 'interpolated_INC':
        # Initialise the etas list
        etas_list : list[float] = [0.0, 3 /4 - 1 / (2 * d), 1.0]
        # Generate a list of random unitary gates and Hamiltonians
        super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_jobs_gates, batch_size = batch_size_gates, is_super = True, etas = etas_list)
        # Generate the QFT/ Chrestenson/ Hadamard gate
        super_gates2, hamiltonians2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = True)
        # Add the two lists together
        super_gates = super_gates + super_gates2
        hamiltonians = hamiltonians + hamiltonians2
    # Generate Haar random gates
    elif method == 'haar':
        # Generate a list of random unitary gates and Hamiltonians
        super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs_gates, batch_size = batch_size_gates, is_super = True)

    # Store the super gates and Hamiltonians for the current dimension
    super_gates_d.append(super_gates)
    hamiltonians_d.append(hamiltonians)    

    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - initialization_time}")

    # Loop over number of gates
    for idx, gate in enumerate(super_gates):

        # GAMMA FIDELITY START TIME
        gamma_fidelity_start_time : float = time.time()

        # Parallelize the gammas loop
        AGF = Parallel(n_jobs = n_jobs_fidelities, batch_size = batch_size_fidelities, verbose = parallel_verbosity)(delayed(compute_fidelity)(hamiltonians[idx], gate, L, times, d, g, options_mesolve) for g in gammas) # type : ignore
        AGI = 1 - np.array(AGF)
        AGIs[idx] = AGI  

        AGIs_1_relative_error[idx] = (AGIs[idx] + AGI_first_order) / AGIs[idx]     

    AGIs_d.append(AGIs)

    AGIs_d_relative_error.append(AGIs_1_relative_error)

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
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gammas = n_gammas, n_gates = n_gates, times = times, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d, AGIs_d_relative_error = AGIs_d_relative_error)

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------