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

n_old_tasks = 2
task_id += n_old_tasks

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()

# Set the figure number
figure_number : int = 10
# Set the method for generating gates
method : str = 'interpolated_INC'
# Set dimensions for the qudits
dims : list[int] = [2, 4]
# Set number of dimensions
n_dims : int = len(dims)
# Set max dimension
d_max : int = max(dims)
# Set collapse operator
collapse_operator : str = "z" # ['x', 'y','z']
# Set alias for tasks
n_tasks : int = task_count
# Set the number of gates
n_gates : int = 4 # 10
# Define a range of noise strengths for the collapse operator
gamma_min : float = 0.0
gamma_max : float = 0.1
gamma_num : float = 41
gammas_ndarray : np.ndarray = np.linspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
gammas : list[float] = gammas_ndarray.tolist()
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set times for the main figure
times : float = 1.0
# Set number of times for the main figure
n_times : int = 1
# Set maximum value of s for the integration, and error threshold
s_max : int = 60
# Set error threshold
error_threshold : float = 1e-8
# Set parallelization parameters
parallel_verbosity : int = 11
# Set the operator flag for the second order AGI calculation
is_super : bool = True
# Set operator type for the second order AGI calculation
AGI_operator_type : str = "SUPER" if is_super else "OPERATOR"
# Gate generation parallelization parameters
n_jobs_gate_generation : int = min(n_CPUs, n_gates)
batch_size_gate_generation : int = int(n_gates / n_jobs_gate_generation)
# Gammas parallelization parameters
n_jobs_gammas : int = min(n_CPUs, n_gammas)
batch_size_gammas : int = int(n_gammas / n_jobs_gammas)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
AGIs_d = []
AGIs_d_first_order = []
AGIs_d_second_order = []
AGIs_d_relative_error = []
hamiltonians_d : list = []
super_gates_d : list = []

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
print (f"QUDIT DIMENSIONS : {dims}")
print(f"GAMMAS : {gammas}")
print(f"MAIN FIGURE TIMES : {times}")
print('\n')
print(f"NUMBER OF GATES : {n_gates}")
print(f"NUMBER OF GAMMAS : {n_gammas}")
print(f"NUMBER OF TIMES FOR MAIN FIGURE : {n_times}")
print(f"NUMBER OF DIMENSIONS : {n_dims}")
print('\n')
print(f"SECOND ORDER AGI CALCULATION METHOD : {AGI_operator_type}")
print(f"MAXIMUM VALUE OF s : {s_max}")
print(f"ERROR THRESHOLD : {error_threshold}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS FOR GATE GENERATION : {n_jobs_gate_generation}")
print(f"BATCH SIZE PER WORKER FOR GATE GENERATION : {batch_size_gate_generation}")
print(f"NUMBER OF PARALLEL JOBS FOR MAIN FIGURE TIMES : {n_jobs_gammas}")
print(f"BATCH SIZE PER WORKER FOR MAIN FIGURE TIMES : {batch_size_gammas}")
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

    if d == 2:
        # Set options for the mesolve function (solver options for differential equations)
        options_mesolve : qt.Options = qt.Options()
        options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
        options_mesolve.max_step = float(1e-8) # type: ignore 
        options_mesolve.nsteps = float(1e9) # type: ignore
        options_mesolve.rtol = float(1e-16)  # Relative tolerance
        options_mesolve.atol = float(1e-16)  # Absolute tolerance
    elif d == 4:
        # Set options for the mesolve function (solver options for differential equations)
        options_mesolve : qt.Options = qt.Options()
        options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
        options_mesolve.max_step = float(0.001) # type: ignore 
        options_mesolve.nsteps = float(1000000) # type: ignore
        options_mesolve.rtol = float(1e-8)  # Relative tolerance
        options_mesolve.atol = float(1e-8)  # Absolute tolerance

    # Initialise a qudit of dimension d
    qudit = Qudit(d = d)
    # Get collapse operators related to the qudit
    j = qudit.get_j
    # Assign collapse operator to L
    L : qt.Qobj = j(collapse_operator) # type:ignore

   # Initialise the etas list for the interpolated INC gates
    etas_list = []
    if method == 'interpolated_INC':
        etas_list = [0.0, (3 * d - 2) / (4 * d), 1.0]
        print(f"ETAS LIST : {etas_list}")

    # Pre-allocate space for the AGI calculations
    AGF : np.ndarray = np.empty(n_gammas, dtype = np.float64)
    AGIs : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    AGI_first_order : np.ndarray = np.empty(n_gammas, dtype = np.float64)    
    AGI_second_order : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    AGI_correction : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    AGIs_relative_error : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")  

    # Generate a list of gates and hamiltonians for the current dimension and method
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_jobs_gate_generation, batch_size = batch_size_gate_generation, is_super = True, etas = etas_list)
    # Generate 1 gate and hamiltonian for the superposition (Hadamard/ Chrestenson/ QFT) gate
    super_gates_2, hamiltonians_2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = True)

    # Concatenate the gates
    super_gates = super_gates + super_gates_2
    # Append the super_gates to the list per dimension
    super_gates_d.append(super_gates)
    # Concatenate the hamiltonians
    hamiltonians = hamiltonians + hamiltonians_2
    # Append the hamiltonians to the list per dimension
    hamiltonians_d.append(hamiltonians)
    
    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - loop_initialization_time}")

    AGI_first_order = (gammas_ndarray * times) * first_order_AGI(d, L)

    # Loop over number of gates
    for gate_idx, gate in enumerate(super_gates):

        print(f"GATE : {gate_idx + 1:03} / {n_gates:03}")

        # Parallelize the gammas loop
        AGF = Parallel(n_jobs = n_jobs_gammas, batch_size = batch_size_gammas, return_as = "list", verbose = parallel_verbosity)(delayed(compute_fidelity)(hamiltonians[gate_idx], gate, L, times, d, g, options_mesolve) for g in gammas) # type : ignore
        # Append the average gate infidelities to the AGI array
        AGIs[gate_idx] = 1 - np.array(AGF)
        # Compute the second order AGI commmutator sum
        comm_sum_second_order, s_finals = second_order_AGI(hamiltonians[gate_idx], L, d, times, s_max, error_threshold = error_threshold, super = is_super)
        # Compute the second order AGI correction
        AGI_second_order[gate_idx] = (gammas_ndarray * times)**2 * comm_sum_second_order
        # Compute the AGI correction for first and second order
        AGI_correction[gate_idx] = AGI_first_order + AGI_second_order[gate_idx]
        # Compute the relative error
        AGIs_relative_error[gate_idx] = (AGIs[gate_idx] + AGI_correction[gate_idx]) / AGIs[gate_idx]

    # Append the AGI and relative error to the lists
    AGIs_d.append(AGIs)
    AGIs_d_first_order.append(AGI_first_order)
    AGIs_d_second_order.append(AGI_second_order)
    AGIs_d_relative_error.append(AGIs_relative_error)

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
    save_data(figure_number, task_id, dims = dims, n_gates = n_gates, hamiltonians_d = hamiltonians_d, times = times, AGIs_d = AGIs_d, AGIs_d_relative_error = AGIs_d_relative_error, gammas = gammas, AGIs_d_first_order = AGIs_d_first_order, AGIs_d_second_order = AGIs_d_second_order)

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------