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
script_file_name : str = "paper_figure_05.py"
# Get SLURM data if running on the cluster
task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)
# Set the print function to flush the buffer (used for parallel printing in SLURM environment)
# Set existing number of data files
n_tasks = 20
task_id += n_tasks
# Set the print function to flush the buffer (used for parallel printing in SLURM environment)
print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()
# Convert start time to real-world time
real_world_start_time : str = time.strftime("%y-%m-%d_%X_%Z").replace("/", "-").replace(":", "-")

# Set the figure number
figure_number : int = 5
# Set the method for generating gates
method : str = 'interpolated_INC'
# Set max dimension for the qudits
d_max : int = 12
# Set dimensions for the qudits
dims : list[int] = np.arange(2, d_max + 1, 1).tolist()
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-2
gamma_num : int = 1 * n_CPUs
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set list of max gamma values
gamma_max_list : list[float] = [1e4]*4 + [1e3]*7 + [1e2]*4
# Set collapse operator
collapse_operator : str = "z" # ['x', 'y','z']
# Set number of random quantum gates
n_gates : int = 25 # 6
etas_list : list = np.linspace(0, 1, n_gates).tolist()
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Error threshold and tolerance for root finding the zeros of the gate infidelity
g_step_size : float = 1e+3
x_tolerance : float = 1e-3
y_threshold : float = 1e-3
# Set parallelization parameters
parallel_verbosity : int = 11
n_jobs_gates : int = min(n_CPUs, n_gates)
batch_size_gates : int = int(n_gates / n_jobs_gates)
n_jobs_fidelities : int = min(n_CPUs, n_gammas)
batch_size_fidelities : int = int(n_gammas / n_jobs_fidelities)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance
# Set list of options.mesolve nsteps values
nsteps_list : list[float] = [1e6]*5 + [1e7]*5 + [1e8]*5

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
print(f'ETAS LIST : {etas_list}')
print(f"GATE TIMES : {times}")
print(f"ROOT FINDING STEP SIZE GAMMA : {g_step_size}")
print(f"ERROR THRESHOLD : {y_threshold}")
print(f"X TOLERANCE : {x_tolerance}")
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
root_results_d : list = []
peak_results_d : list = []
plateau_AGIs_d : list = []
plateau_gammas_d : list = []

# INITIALIZATION TIME
initialization_time : float = time.time()
print(f"INITIALISATION TIME: {initialization_time - start_time:.4f}")

print('\n')

# Loop over the dimensions
for d_idx, d in enumerate(dims):

    # LOOP START TIME
    loop_start_time : float = time.time()  
    
    print(f"DIMENSION : {d:02} / {d_max:02}")

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

    # Initialise the etas list
    # etas_list = (3 * d - 2) / (4 * d)
    # etas_list = np.linspace(0, etas_list, n_gates - 3)
    # etas_list = np.append(etas_list, (etas_list[-1] + etas_list[-2]) / 2)
    # etas_list = np.append(etas_list, 1)
    # etas_list = np.sort(etas_list).tolist()

    # Pre-allocate space for local data for the current dimension
    AGIs : np.ndarray = np.zeros((n_gates, n_gammas), dtype = np.float64) 
    plateau_AGIs : np.ndarray = np.zeros(n_gates, dtype = np.float64)
    plateau_gammas : np.ndarray = np.zeros(n_gates, dtype = np.float64)

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time:.4f}")   

    # Generate a list of random unitary matrices
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, d, method = method, n_jobs = n_jobs_gates, batch_size = batch_size_gates, is_super = True, etas = etas_list)
    # Generate the QFT/ Chrestenson/ Hadamard gate
    # super_gates2, hamiltonians2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = True)
    # # Add the two lists together
    # super_gates = super_gates + super_gates2
    # hamiltonians = hamiltonians + hamiltonians2
    # Store the super gates and Hamiltonians for the current dimension
    hamiltonians_d.append(hamiltonians)
    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - loop_initialization_time:.4f}")

    # Prepare parallel computation tasks
    tasks = [(gamma_idx, gate_idx, hamiltonians[gate_idx], gate, L, times, d, gamma, options_mesolve) for gamma_idx, gamma in enumerate(gammas) for gate_idx, gate in enumerate(super_gates)]
    # Execute tasks in parallel
    fidelity_results = Parallel(n_jobs = n_CPUs, batch_size='auto', verbose=parallel_verbosity)(delayed(compute_fidelity_wrapper)(*task) for task in tasks)    
    # Collect results
    for gamma_idx, gate_idx, fidelities in fidelity_results:
        AGIs[gate_idx, gamma_idx] = 1 - fidelities
    AGIs_d.append(AGIs)
    # FIDELITY TIME
    fidelity_time = time.time()
    print(f"Fidelity time: {fidelity_time - gate_generation_time:.4f}")

    # # Find the roots of the AGI curves
    # root_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(find_AGI_plateau_roots)(AGIs[n], gammas_array, y_threshold, index = n) for n in range(n_gates)) # type : ignore
    # root_results_d.append(root_results)

    # # Find the peak values of the AGI curves
    # peak_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(find_AGI_peak)(AGIs[n], gammas_array, index = n) for n in range(n_gates)) # type : ignore
    # peak_results_d.append(peak_results)

    # ROOT FINDING METHOD
    # Loop over number of gates
    for idx, gate in enumerate(super_gates): 
        # Fit cubic spline
        spline = CubicSpline(gammas, AGIs[idx], bc_type='natural')
        # Function to find roots of
        def func_to_solve(x, threshold : float = 1e-3):
            return np.abs(spline(x) - spline(gamma_max)) - threshold
        # Initial guess
        x_guess = 1.0
        root = fsolve(func_to_solve, x_guess, xtol = x_tolerance)
        plateau_gammas[idx] = root[0]
        plateau_AGIs[idx] = spline(root[0])

    plateau_gammas_d.append(plateau_gammas)
    plateau_AGIs_d.append(plateau_AGIs)

    # ROOT FINDING TIME
    root_finding_time = time.time()
    print(f"Root finding time: {root_finding_time - fidelity_time:.4f} seconds")    

    # Perform garbage collection to free up memory
    gc.collect()
    # Clear large variables to free up memory
    del qudit, super_gates, hamiltonians, AGIs, tasks, fidelity_results, spline, root, plateau_gammas, plateau_AGIs, tasks, fidelity_results, gamma_idx, gate_idx, fidelities
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
    # NEW METHOD
    # save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gates = n_gates, times = times, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d, root_results_d = root_results_d, peak_results_d = peak_results_d)  
    # OLD METHOD  
    save_data(figure_number, task_id, dims = dims, etas_list = etas_list, gammas = gammas, n_gates = n_gates, times = times, AGIs_d = AGIs_d, plateau_gammas_d = plateau_gammas_d, plateau_AGIs_d = plateau_AGIs_d)    
# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time:.4f}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time:.4f}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------