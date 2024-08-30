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
script_file_name : str = "paper_figure_04.py"

task_count, task_id, n_CPUs, figure_number = get_SLURM_Data(script_file_name)

n_tasks = 1
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
figure_number : int = 4
# Set the method for generating gates
method : str = 'interpolated_INC'
# Set collapse operator
collapse_operator : str = 'z'
# Set dimensions for the qudits
dims : list[int] = [4]
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-4
gamma_max : float = 1e+4
gamma_num : int = 2 * n_CPUs #50
gammas_ndarray : np.ndarray = np.geomspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
gammas : list[float] = gammas_ndarray.tolist()
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set number of random quantum gates
n_gates : int = 6
# Set number of random initial density matrix states
n_states : int = 100 * n_CPUs
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Set parallelization parameters
parallel_verbosity : int = 11
n_jobs_gates : int = min(n_CPUs, n_gates)
batch_size_gates : int = int(n_gates / n_jobs_gates)
n_jobs_gammas : int = int(n_CPUs)
batch_size_gammas : int = int(n_gammas / n_jobs_gammas)   
n_jobs_states : int = int(n_CPUs)
batch_size_states : int = int(n_states / n_jobs_states)
# Set options for the mesolve function (solver options for differential equations)
options_mesolve : qt.Options = qt.Options()
options_mesolve.method = 'bdf'  # Setting method to 'bdf' (backward differentiation formula)
options_mesolve.max_step = float(0.001) # type: ignore 
options_mesolve.nsteps = float(1000000) # type: ignore
options_mesolve.rtol = float(1e-8)  # Relative tolerance
options_mesolve.atol = float(1e-8)  # Absolute tolerance

# Pre-allocate space for data
AGIs_d : list = []
coherences_d : list = []
super_gates_d : list = []
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
print(f"GAMMAS : {gammas}")
print(f"GATE TIMES : {times}")
print('\n')
print(f"PARALLEL VERBOSITY : {parallel_verbosity}")
print(f"NUMBER OF PARALLEL JOBS FOR FIDELITIES : {n_jobs_gammas}")
print(f"BATCH SIZE PER FIDELITY JOB : {batch_size_gammas}")
print(f"NUMBER OF PARALLEL JOBS FOR INITIAL STATES : {n_jobs_states}")
print(f"BATCH SIZE PER STATE JOB : {batch_size_states}")

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

    # Pre-allocate space for local data
    AGIs : np.ndarray = np.empty((n_gates, n_gammas), dtype = np.float64)
    AGF : np.ndarray = np.empty(n_gammas, dtype = np.float64)
    final_state : list = [qt.Qobj(np.empty((d, d), dtype = np.complex128)) for _ in range(n_states)]
    final_states_g : list = []
    coherences_gates : list = []
    purities_gates : list = []
    entropies_gates : list = []

    # Initialize a qudit of dimension d
    qudit = Qudit(d = d)
    # Get qudit collapse operators
    j = qudit.get_j
    L : qt.Qobj = j(collapse_operator) # type: ignore         

    # Define random initial states
    initial_kets = [qt.rand_ket(d) for _ in range(n_states)]
    initial_states = [initial_kets[n] * initial_kets[n].dag() for n in range(n_states)]

    # Initialise the etas list
    etas_list = (3 * d - 2) / (4 * d)
    etas_list = np.linspace(0, etas_list, n_gates - 3)
    etas_list = np.append(etas_list, (etas_list[-1] + etas_list[-2]) / 2)
    etas_list = np.append(etas_list, 1)    
    etas_list = np.sort(etas_list).tolist()    

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")

    # Generate a list of random unitary gates and Hamiltonians
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_jobs_gates, batch_size = batch_size_gates, is_super = True, etas = etas_list)
    # Generate the QFT/ Chrestenson/ Hadamard gate
    super_gates2, hamiltonians2 = generate_gates_and_hamiltonians(1, d, method = 'QFT', n_jobs = 1, batch_size = 1, is_super = True)

    # Add the two lists together
    super_gates = super_gates + super_gates2
    hamiltonians = hamiltonians + hamiltonians2

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

        print(f'LOOP OVER GATE : {idx + 1:03} / {n_gates:03}')

        # Parallelize the gammas loop for the fidelities
        print(f'PARALLELISED AGFs FOR : {n_gammas: 03} GAMMAS')
        AGF = Parallel(n_jobs = n_jobs_gammas, batch_size = batch_size_gammas, verbose = parallel_verbosity)(delayed(compute_fidelity)(hamiltonians[idx], gate, L, times, d, g, options_mesolve) for g in gammas) # type : ignore             

        # Pre-allocate space for the final states, coherences, purities and entropies
        final_states = np.zeros((n_states, n_gammas), dtype = qt.Qobj) 
        coherences = np.zeros((n_states, n_gammas), dtype = np.float64)
        purities = np.zeros((n_states, n_gammas), dtype = np.float64)
        entropies = np.zeros((n_states, n_gammas), dtype = np.float64)

        # Loop over the number of random initial states
        for g_idx, g in enumerate(gammas):
            # DENSITY MATRIX START TIME
            density_matrix_start_time : float = time.time()

            print(f'LOOP OVER GAMMA : {g_idx + 1:02} / {n_gammas:02}')

            # Parallelize the gammas loop for the coherences
            print(f'PARALLELISED INITIAL STATES')
            final_state = Parallel(n_jobs = n_jobs_states, batch_size = batch_size_states, verbose = parallel_verbosity)(delayed(propagate_density_matrix)(hamiltonians[idx], initial_states[s], L, times, d, g, options_mesolve) for s in range(n_states)) # type : ignore
            print('PARALLELISED COHERENCES')
            coherences[:, g_idx] = Parallel(n_jobs = n_jobs_states, batch_size = batch_size_states, verbose = parallel_verbosity)(delayed(average_coherences_density_matrix)(final_state[s]) for s in range(n_states)) # type : ignore
            print('PARALLELISED PURITIES')
            purities[:, g_idx] = Parallel(n_jobs = n_jobs_states, batch_size = batch_size_states, verbose = parallel_verbosity)(delayed(purity_density_matrix)(final_state[s], d) for s in range(n_states)) # type : ignore
            print('PARALLELISED ENTROPIES')
            entropies[:, g_idx] = Parallel(n_jobs = n_jobs_states, batch_size = batch_size_states, verbose = parallel_verbosity)(delayed(von_neumann_entropy)(final_state[s]) for s in range(n_states)) # type : ignore
            
            # Store the final states for the current initial state
            final_states[:, g_idx] = final_state

            # DENSITY MATRIX TIME
            density_matrix_time : float = time.time()
            print(f'{n_states:04} FINAL STATES FOR GAMMA {g_idx + 1:02} / {n_gammas:02} completed in: {density_matrix_time - density_matrix_start_time}') 
            print('\n')       

        # GAMMA FIDELITY TIME
        gamma_fidelity_time : float = time.time()
        print(f"Gate AGFs & DMs: {idx + 1:03} / {n_gates:03} completed in: {gamma_fidelity_time - gamma_fidelity_start_time}")
        
        # Store the final states for the current gate
        final_states_g.append(final_states)

        # Aggregate the properties of the final density matrices
        coherences_gates.append([np.mean(coherences[:, g_idx]) for g_idx in range(n_gammas)])
        purities_gates.append([np.mean(purities[:, g_idx]) for g_idx in range(n_gammas)])
        entropies_gates.append([np.mean(entropies[:, g_idx]) for g_idx in range(n_gammas)])
        
        # Store the AGI and average coherence data for the current gate
        AGIs[idx] = 1 - np.array(AGF)

    # Store the final data for the current dimension
    AGIs_d.append(AGIs)
    final_states_d.append(final_states_g)
    coherences_d.append(coherences_gates)
    purities_d.append(purities_gates)
    entropies_d.append(entropies_gates)

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
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gammas = n_gammas, n_gates = n_gates, n_states = n_states, times = times, AGIs_d = AGIs_d, hamiltonians_d = hamiltonians_d, final_states_d = final_states_d, coherences_d = coherences_d, purities_d = purities_d, entropies_d = entropies_d)
 

# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------