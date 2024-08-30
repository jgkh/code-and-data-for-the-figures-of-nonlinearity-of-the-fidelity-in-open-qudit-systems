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
script_file_name : str = "paper_figure_14.py"

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

n_tasks = 5
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
figure_number : int = 14
# Set the method for generating gates
method : str = 'interpolated_INC'
# Set collapse operator
collapse_operator : str = 'z'
# Set dimensions for the qudits
dims : list[int] = [2, 4, 8, 16]
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-4
gamma_max : float = 1e+3
gamma_num : int = 50
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set list of max gamma values
gamma_max_list : list[float] = [1e4]*2 + [1e3]*1 + [1e2]*1
# Set number of random quantum gates
n_gates : int = 6
# Set number of random initial density matrix states
n_states : int = 100
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Set parallelization parameters
parallel_verbosity : int = 11
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance
# Set list of options.mesolve nsteps values
nsteps_list : list[float] = [1e6]*2 + [1e7]*1 + [1e8]*1

# Pre-allocate space for data
AGIs_d : list = []
hamiltonians_d : list = []
final_states_d : list = []
coherences_d : list = []
purities_d : list = []
entropies_d : list = []

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
print(f"GATE TIMES : {times}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
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
    
    print(f"DIMENSION: {d}")

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

    # Initialise a qudit of dimension d
    qudit = Qudit(d = d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j(collapse_operator) # type: ignore         

    # Initialise random initial states
    initial_kets = [qt.rand_ket(d) for _ in range(n_states)]
    initial_states = [initial_kets[n] * initial_kets[n].dag() for n in range(n_states)]

    # Initialise the etas list
    etas_list = (3 * d - 2) / (4 * d)
    etas_list = np.linspace(0, etas_list, n_gates - 3)
    etas_list = np.append(etas_list, (etas_list[-1] + etas_list[-2]) / 2)
    etas_list = np.append(etas_list, 1)    
    etas_list = np.sort(etas_list).tolist()    

    # LOOP INITIALISATION TIME
    loop_initialisation_time : float = time.time()
    print(f"Loop initialisation time: {loop_initialisation_time - loop_start_time}")

    # GATE GENERATION START
    # Generate a list of random unitary gates and Hamiltonians
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_CPUs, batch_size = 1, is_super = True, etas = etas_list)
    # Generate the QFT/ Chrestenson/ Hadamard gate
    super_gates2, hamiltonians2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = True)
    # Add the two lists together
    super_gates = super_gates + super_gates2
    hamiltonians = hamiltonians + hamiltonians2
    # Store the super gates and Hamiltonians for the current dimension
    hamiltonians_d.append(hamiltonians)    
    # GATE GENERATION TIME
    gate_generation_time = time.time()
    print(f"Gate generation time: {gate_generation_time - initialization_time}")

    # FIDELITY START
    # Initialise AGIs array
    AGIs : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    # Prepare parallel computation tasks
    fidelity_tasks = [(gamma_idx, gate_idx, hamiltonians[gate_idx], gate, L, times, d, gamma, options_mesolve) for gamma_idx, gamma in enumerate(gammas) for gate_idx, gate in enumerate(super_gates)]
    # Execute tasks in parallel
    fidelity_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(compute_fidelity_wrapper)(*task) for task in fidelity_tasks)
    # Collect results
    for gamma_idx, gate_idx, fidelities in fidelity_results: # type : ignore
        AGIs[gate_idx, gamma_idx] = 1 - fidelities
    AGIs_d.append(AGIs)
    # FIDELITY TIME
    fidelity_time = time.time()
    print(f"Fidelity time: {fidelity_time - gate_generation_time:.4f}")

    # DENSITY MATRIX START
    # Initialise the final_states array
    final_states : np.ndarray = np.zeros((n_gates, n_states, n_gammas), dtype = qt.Qobj)
    # Prepare parallel computation tasks
    density_matrix_tasks = [(gate_idx, state_idx, gamma_idx, hamiltonians[gate_idx], state, L, times, d, gamma, options_mesolve) for gamma_idx, gamma in enumerate(gammas) for state_idx, state in enumerate(initial_states) for gate_idx, gate in enumerate(super_gates)]
    # Execute tasks in parallel
    density_matrix_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(propagate_density_matrix_wrapper)(*task) for task in density_matrix_tasks)  
    # Collect results
    for gate_idx, state_idx, gamma_idx, density_matrix in density_matrix_results: # type : ignore
        final_states[gate_idx, state_idx, gamma_idx] = density_matrix
    final_states_d.append(final_states)
    # DENSITY MATRIX TIME
    density_matrix_time : float = time.time()
    print(f"Density matrix time: {density_matrix_time - fidelity_time:.4f}")

    # AVERAGE COHERENCES START
    # Initialise the average_coherences array
    average_coherences : np.ndarray = np.zeros((n_gates, n_states, n_gammas), dtype = np.float64)
    # Prepare parallel computation tasks
    average_coherences_tasks = [(gate_idx, state_idx, gamma_idx, final_states[gate_idx, state_idx, gamma_idx]) for gamma_idx in range(n_gammas) for state_idx in range(n_states) for gate_idx in range(n_gates)]
    # Execute tasks in parallel
    average_coherences_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(average_coherences_density_matrix_wrapper)(*task) for task in average_coherences_tasks)  
    # Collect results
    for gate_idx, state_idx, gamma_idx, average_coherence in average_coherences_results: # type : ignore
        average_coherences[gate_idx, state_idx, gamma_idx] = average_coherence
    coherences_d.append(average_coherences)
    # AVERAGE COHERENCES TIME
    average_coherences_time : float = time.time()
    print(f"Average coherences time: {average_coherences_time - density_matrix_time:.4f}")

    # STATE PURITY START
    # Initialise the state_purities array
    state_purities : np.ndarray = np.zeros((n_gates, n_states, n_gammas), dtype = np.float64)
    # Prepare parallel computation tasks
    state_purities_tasks = [(gate_idx, state_idx, gamma_idx, final_states[gate_idx, state_idx, gamma_idx], d) for gamma_idx in range(n_gammas) for state_idx in range(n_states) for gate_idx in range(n_gates)]
    # Execute tasks in parallel
    state_purities_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(purity_density_matrix_wrapper)(*task) for task in state_purities_tasks)
    # Collect results
    for gate_idx, state_idx, gamma_idx, state_purity in state_purities_results: # type : ignore
        state_purities[gate_idx, state_idx, gamma_idx] = state_purity
    purities_d.append(state_purities)
    # STATE PURITY TIME
    state_purity_time : float = time.time()
    print(f"State purity time: {state_purity_time - average_coherences_time:.4f}")

    # VON NEUMANN ENTROPY START
    # Initialise the von_neumann_entropies array
    von_neumann_entropies : np.ndarray = np.zeros((n_gates, n_states, n_gammas), dtype = np.float64)
    # Prepare parallel computation tasks
    von_neumann_entropies_tasks = [(gate_idx, state_idx, gamma_idx, final_states[gate_idx, state_idx, gamma_idx]) for gamma_idx in range(n_gammas) for state_idx in range(n_states) for gate_idx in range(n_gates)]
    # Execute tasks in parallel
    von_neumann_entropies_results = Parallel(n_jobs = n_CPUs, batch_size = 'auto', verbose = parallel_verbosity)(delayed(von_neumann_entropy_wrapper)(*task) for task in von_neumann_entropies_tasks)
    # Collect results
    for gate_idx, state_idx, gamma_idx, entropy in von_neumann_entropies_results: # type : ignore
        von_neumann_entropies[gate_idx, state_idx, gamma_idx] = entropy
    entropies_d.append(von_neumann_entropies)
    # VON NEUMANN ENTROPY TIME
    von_neumann_entropy_time : float = time.time()
    print(f"Von Neumann entropy time: {von_neumann_entropy_time - state_purity_time:.4f}")

    # GARBAGE COLLECTION
    gc.collect()
    del qudit, initial_kets, initial_states, etas_list, super_gates, super_gates2, hamiltonians, hamiltonians2, AGIs, fidelity_tasks, fidelity_results, final_states, density_matrix_tasks, density_matrix_results, average_coherences, average_coherences_tasks, average_coherences_results, state_purities, state_purities_tasks, state_purities_results, von_neumann_entropies, von_neumann_entropies_tasks, von_neumann_entropies_results
    gc.collect()

    # SINGLE LOOP TIME
    loop_time = time.time()
    print(f"Single loop time: {loop_time - loop_start_time:.4f}")

    print('\n')    

# TOTAL LOOP TIME
total_loop_time = time.time()
print(f"Total loop time: {total_loop_time - start_time:.4f}")

# Save the data
if save_figure_data:
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gammas = n_gammas, n_gates = n_gates, n_states = n_states, times = times, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d, final_states_d = final_states_d, coherences_d = coherences_d, purities_d = purities_d, entropies_d = entropies_d)
 

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time:.4f}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time:.4f}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------