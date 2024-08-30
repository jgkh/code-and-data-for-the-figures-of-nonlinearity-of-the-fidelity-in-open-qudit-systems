# ----------------------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------------------

import paper_imports
import paper_methods

from paper_imports import *
from paper_methods import *

importlib.reload(paper_imports)
importlib.reload(paper_methods)

# ----------------------------------------------------------------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------------------------------------------------------------

# Save data to file
save_figure_data : bool = False

# ----------------------------------------------------------------------------------------------------------------------------------
# HPC SETTINGS
# ----------------------------------------------------------------------------------------------------------------------------------

try:
    
    task_count : int = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    
except:
    
    task_count : int = 1
try:
    
    task_id : int = int(os.environ['SLURM_ARRAY_TASK_ID'])   
    
except:
    
    task_id : int = 1

try:
    
    n_CPUs : int = int(os.environ["SLURM_CPUS_ON_NODE"])
    
except:
    
    n_CPUs : int = int(os.cpu_count()) # type: ignore

print = functools.partial(print, flush = True)

# ----------------------------------------------------------------------------------------------------------------------------------
# START OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------

# START TIMING
start_time : float = time.time()

# Set the figure number
figure_number : int = 6
# Set the method for generating gates
method : str = 'interpolated_INC'
# Set dimensions for the qudits
dims : list[int] = [2, 4]
# Set the range of noise strengths for the collapse operator
gamma_min : float = 1e-3
gamma_max : float = 1e+3
gamma_num : int = 50
gammas_ndarray : np.ndarray = np.geomspace(gamma_min, gamma_max, gamma_num, dtype = np.float64)
gammas : list[float] = gammas_ndarray.tolist()
# Set alias for gamma_num for readability
n_gammas : int = gamma_num
# Set number of random quantum gates
n_gates : int = 4
# Set parallelization parameters
n_jobs : int = min(n_CPUs, n_gates)
batch_size : int = int(n_gates / n_jobs)   
# Set times for the propagator; currently set to just one value, 1.0
times : float = 1.0
# Set number of random initial density matrix states
n_states = 1
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
    L : qt.Qobj = j("x") # type: ignore      

    # Define random initial states
    initial_kets = [qt.rand_ket(d) for _ in range(n_states)]
    initial_states = [initial_kets[n] * initial_kets[n].dag() for n in range(n_states)]

    # Initialise the etas list
    etas_list : list[float] = [0.0, 3 /4 - 1 / (2 * d), 1.0]

    # Pre-allocate space for local data for the current dimension
    AGIs = np.empty((n_gates, n_gammas), dtype = np.float64)
    coherences = np.empty((n_states, n_gammas), dtype = np.float64)
    AGF = np.empty(n_gammas, dtype = np.float64)
    average_coherence = np.empty(n_gammas, dtype = np.float64)
    mean_coherences = np.empty((n_gates, n_gammas), dtype = np.float64)

    final_state = [qt.Qobj(np.empty((d, d), dtype = np.complex128)) for _ in range(n_gammas)]
    final_states = []
    final_states_g = []

    # LOOP INITIALIZATION TIME
    loop_initialization_time : float = time.time()
    print(f"Loop initialization time: {loop_initialization_time - loop_start_time}")    

    # Generate a list of random unitary gates and Hamiltonians
    super_gates, hamiltonians = generate_gates_and_hamiltonians(n_gates - 1, d, method = method, n_jobs = n_jobs, batch_size = batch_size, is_super = True, etas = etas_list)
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

        # Parallelize the gammas loop for the fidelities
        AGF = Parallel(n_jobs = n_jobs, batch_size = batch_size)(delayed(compute_fidelity)(hamiltonians[idx], gate, L, times, d, g, options_mesolve) for g in gammas) # type : ignore      

        final_states = [] 

        # Loop over the number of random initial states
        for n in range(n_states):
            # DENSITY MATRIX START TIME
            density_matrix_start_time : float = time.time()
            # Parallelize the gammas loop for the coherences
            # coherences[n] = Parallel(n_jobs = n_jobs, batch_size = batch_size)(delayed(compute_coherences)(hamiltonians[idx], initial_states[n], L, times, d, g, options_mesolve) for g in gammas) # type : ignore
            final_state = Parallel(n_jobs = n_jobs, batch_size = batch_size)(delayed(propagate_density_matrix)(hamiltonians[idx], initial_states[n], L, times, d, g, options_mesolve) for g in gammas) # type : ignore
            # Store the final states for the current initial state
            final_states.append(final_state)

            # DENSITY MATRIX TIME
            density_matrix_time : float = time.time()
            print(f'State {n + 1:02} / {n_states:02} completed in: {density_matrix_time - density_matrix_start_time}')

        # GAMMA FIDELITY TIME
        gamma_fidelity_time : float = time.time()
        print(f"Gate AGFs & DMs: {idx + 1:03} / {n_gates:03} completed in: {gamma_fidelity_time - gamma_fidelity_start_time}")

        # Calculate the average coherence for each gamma value over all the random initial states
        # average_coherence = np.mean(coherences, axis = 0)
        
        final_states_g.append(final_states)
        
        # Store the AGI and average coherence data for the current gate
        AGIs[idx] = 1 - np.array(AGF)
        # mean_coherences[idx] = np.array(average_coherence)

    # Store the AGI data per dimension
    AGIs_d.append(AGIs)
    # Store the mean coherence data per dimension
    # coherences_d.append(mean_coherences)
    final_states_d.append(final_states_g)

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
    save_data(figure_number, task_id, dims = dims, gammas = gammas, n_gammas = n_gammas, n_gates = n_gates, n_states = n_states, times = times, AGIs_d = AGIs_d, coherences_d = coherences_d, hamiltonians_d = hamiltonians_d, super_gates_d = super_gates_d, final_states_d = final_states_d)


# SAVE DATA TIME
save_data_time = time.time()
print(f"Save data time: {save_data_time - total_loop_time}")

# TOTAL TIME
total_time = time.time()
print(f"Total time: {total_time - start_time}")

# ----------------------------------------------------------------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------------------