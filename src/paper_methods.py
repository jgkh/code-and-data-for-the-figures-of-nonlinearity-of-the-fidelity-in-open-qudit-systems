# ----------------------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------------------

import paper_imports

from paper_imports import *

importlib.reload(paper_imports)

import paper_imports

from paper_imports import *

# ----------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def get_SLURM_Data(script_name : str, output_log : bool = True) -> tuple[int, int, int, int]:
    """
    Description:
    ------------
    Get the SLURM data from the environment variables and also output to the console if output_log is True

    Parameters:
    -----------
    output_log : bool (default = True)
        If True, output the SLURM data to the console

    Returns:
    --------
    task_count : int
        The number of tasks in the current job
    task_id : int
        The task ID for the current job
    n_CPUs : int
        The number of workers on the current node
    figure_number : int
        The figure number for the current script

    Raises:
    -------
    TypeError
        If script_name is not a string, or
        If output_log is not a boolean

    FileNotFoundError
        If the file with script_name is not found in the working directory

    Examples:
    ---------
    >>> task_id, n_CPUs, figure_number = get_SLURM_Data(filename)    

    """

    # Check if script_name is a string
    if not isinstance(script_name, str):
        raise TypeError("Parameter script_name must be a string")
    
    # Check if file with script_name exists in the working directory
    if not os.path.isfile(script_name):
        raise FileNotFoundError(f"File {script_name} not found in the working directory")

    # Check if output_log is a boolean
    if not isinstance(output_log, bool):
        raise TypeError("Parameter output_log must be a boolean")

    # Initialise return variables
    task_count : int = 1
    task_id : int = 1
    n_CPUs : int = 1
    figure_number : int = 1

    # Intialise local variables
    job_id : str = 'NOSLURM'
    match : re.Match
    figure_number : int = 0

    try:
        job_id = os.environ["SLURM_JOB_ID"]    
    except:
        job_id = 'NOSLURM'
    
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

    try:
        figure_number : int = int(sys.argv[3])
    except:
        # Regular expression to match the figure number
        match = re.search(r"paper_figure_(\d{2})\.py", script_name) # type: ignore
        figure_number : int = int(match.group(1))

    if output_log:
        # Print the SLURM data
        print(f'FIGURE_NUMBER: {figure_number:02}')
        print(f'SCRIPT_NAME: {script_name}')
        print('\n')
        print(f"SLURM_JOB_ID: {job_id}")
        print(f"SLURM_ARRAY_TASK_COUNT: {task_count}")
        print(f"SLURM_ARRAY_TASK_ID: {task_id}")
        print(f"SLURM_CPUS_ON_NODE: {n_CPUs}")
        print('\n')
    
    return task_count, task_id, n_CPUs, figure_number
# ----------------------------------------------------------------------------------------------------------------------------------
def save_data(figure_number : int, task_number : int = 1, **kwargs) -> None:
    """
    Saves multiple variables to a file using pickle, with the filename automatically generated.
    
    Parameters:
    -----------
    figure_number : int
        The figure number in the paper that the data is associated with.
    task_number : int
        The task number that the data is associated with.
    **kwargs : dict
        The variables to be saved. The keys are the variable names and the values are the variables themselves.
    
    Returns:
    --------
    None
    
    Raises:
    -------
    None

    Examples:
    ---------
    # Save a single variable for figure 2
        save_data(2, AGIs_d=AGIs_d)
    # Save multiple variables for figure 7, task 1
        save_data(7, 1, AGIs_d=AGIs_d, another_variable=another_variable)
    """
    
    # Initialize local variables
    filename = f'../dat/Figure_{figure_number:02}_results_{task_number:03}.blob'

    # Use pickle to save the variables to a file
    with open(filename, 'wb') as file:
        pickle.dump(kwargs, file)
# ----------------------------------------------------------------------------------------------------------------------------------
def load_data(figure_number, task_number : int = 1) -> dict:
    """
    Loads and returns data from a file using pickle. 
    The data is stored and returned as a dictionary.
    The filename is automatically generated based on the figure number.
    
    Parameters:
    -----------
    figure_number : int
        The figure number in the paper that the data is associated with.
    task_number : int
        The task number that the data is associated with.
    
    Returns:
    --------
    dict
        The data loaded from the file.

    Raises:
    -------
    FileNotFoundError
        If the file is not found.

    Examples:
    ---------
    # Load data for figure 2
        data = load_data(2)
        AGIs_d = data['AGIs_d']
        another_variable = data['another_variable']
    """

    # Initialize local variables
    filename = f'../dat/Figure_{figure_number:02}_results_{task_number:03}.blob'

    # Use pickle to load the variables from a file and return them as a dictionary
    with open(filename, 'rb') as file:
        return pickle.load(file)
# ----------------------------------------------------------------------------------------------------------------------------------
def unitary_matrix_logarithm(U : np.ndarray) -> np.ndarray :
    """
    Compute the matrix logarithm of a unitary matrix U using the general Schur decomposition U = Q * T * Q_dag, where Q is unitary and T is diagonal.
    For certain unitaries with eigenvalues near -1, the standard Scipy logm function can fail to produce a skew-Hermitian matrix.
    Algorithm based on the paper in Reference [1].

    Parameters:
    -----------
    U : np.ndarray
        A unitary matrix for which to compute the matrix logarithm.

    Returns:
    --------
    J : np.ndarray
        The skew-Hermitian matrix logarithm of the unitary matrix U.

    Raises:
    -------
    TypeError
        If the input matrix is not a numpy array.
    AssertionError
        If the input matrix is not unitary, or
        If the Schur decomposition fails to produce a unitary matrix Q and a diagonal matrix T.

    Examples:
    ---------
    >>> J = unitary_matrix_logartihm(U)

    References:
    -----------
    [1] Terry A. Loring, "Computing a logarithm of a unitary matrix with general spectrum", Numerical Linear Algebra with Applications, 2014, 21(6), 744-760, https://doi.org/10.1002/nla.1927, https://arxiv.org/abs/1203.6151 
    """

    # Check if the input matrix is a numpy array
    if not isinstance(U, np.ndarray):
        raise TypeError("Parameter U must be a numpy array")

    # Check if the input matrix is unitary
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise AssertionError("Parameter U must be a unitary matrix")
    
    # Initialise return variables
    J : np.ndarray = np.zeros(U.shape, dtype = np.complex128)

    # Initialise local variables
    d : int = U.shape[0]
    Q : np.ndarray = np.zeros((d, d), dtype = np.complex128)
    T : np.ndarray = np.zeros((d, d), dtype = np.complex128)
    log_D : np.ndarray = np.zeros((d, d), dtype = np.complex128)

    # Compute the Schur decomposition of the unitary matrix U = Q * T * Q_dag
    T, Q = sp.linalg.schur(U, output = 'complex')
    # Check if Q is unitary
    assert np.allclose(Q @ Q.conj().T, np.eye(d)), "Error in Schur decomposition"
    # Check if T is diagonal
    assert np.allclose(T, np.diag(T.diagonal())), "Error in Schur decomposition"

    # Compute the diagonal unitary matrix D of the log of normalised eigenvalues of T
    log_D = np.diag(np.log(T.diagonal() / np.abs(T.diagonal())))

    # Compute the skew-Hermitian matrix logarithm of the unitary matrix U
    J = Q @ log_D @ Q.conj().T

    return J
# ----------------------------------------------------------------------------------------------------------------------------------
def find_AGI_plateau_adaptive(H : qt.Qobj, U : qt.Qobj, L : qt.Qobj, t : float, d : int, options_mesolve : qt.Options = qt.Options(), gamma_max : float = 1e4, gamma_step : float = 1e3, y_threshold : float = 1e-3, x_tolerance : float = 1e-3, save_data : bool = False) -> tuple[float, float, list[tuple[float, float]]]:
    """
    Description:
    ------------
    Find the largest gamma and value for a certain gate, U, and hamiltonian, H, such that the AGI curve converges to the plateau value within a certain threshold.
    This corresponds to finding the last instance when the AGI curve (as a function of gamma) lies outside of the threshold region of the AGI plateau value.
    This method uses a reverse adaptive search to find the root, instead of a standard root finding algorithm.
    It leverages the fact that the AGI curve converges to the plateau value monotonically for large values of gamma, and that we are able to calculate the AGI numerically for any given gamma.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.
    U : qt.Qobj
        The gate superoperator.
    L : qt.Qobj
        The Lindbladian collapse operator.
    t : float
        The time of the gate.
    d : int
        The dimension of the system.
    options_mesolve : qt.Options, optional (default = qt.Options())
        The options for the mesolve function.
    gamma_max : float, optional (default = 1e4)
        The maximum value of gamma from which to start the reverse search for the AGI plateau.
    gamma_step : float, optional (default = 1e3)
        The starting step size for the reverse search.
    y_threshold : float, optional (default = 1e-3)
        The threshold region for the AGI plateau value.
    x_tolerance : float, optional (default = 1e-3)
        The tolerance for the root-finding algorithm on the dependent variable gamma.
    save_data : bool, optional (default = False)
        Whether to save the search data for the AGI curve during the reverse search.

    Returns:
    --------
    gamma_root_value : float
        The AGI value at the root.
    AGI_root_value : float
        The gamma value at the root.
    saved_data : list[tuple[float, float]]
        The search data for each gamma and AGI calculation during the reverse search.
    
    Raises:
    -------
    TypeError: Type mismatch between method parameters and input arguments. Possible reasons include:
        - H is not a Qobj.
        - H is not Hermitian.
        - U is not a Qobj.
        - U is not a superoperator.
        - L is not a Qobj.
        - t is not a float.
        - d is not an int.
        - gamma_max is not a float.
        - gamma_step is not a float.
        - y_threshold is not a float.
        - x_tolerance is not a float.
        - options_mesolve is not a qt.Options.
        - save_data is not a bool.

    ValueError: Inappropriate values for input arguments. Possible reasons include:
        - H is not a square matrix of dimension d.
        - U is not a square matrix of dimension d * d.
        - L is not a square matrix of dimension d.
        - t is not positive.
        - d is less than 2.
        - gamma_max is not positive.
        - gamma_step is not positive, or larger than gamma_max.
        - y_threshold is not positive.
        - x_tolerance is not positive.

    UserWarning: Possible issues with the input arguments. Possible reasons include:
        - The search result for gamma lies outside of the range of [0, gamma_max].
        - The search result for AGI lies outside of the range of [0, 1].
        - The search result for AGI does not converge to the plateau value of AGI(gamma_max) within the threshold region.

    See Also:
    ---------
    - script_methods.compute_fidelity

    Examples:
    ---------
    Find the AGI plateau root for the qubit Hadamard (QFT for d = 2) gate.
    Make use of default values for the optional parameters: options_mesolve, gamma_max, gamma_step, y_threshold, x_tolerance, and save_data.
    >>> import qutip as qt
    >>> from script_methods import QFT_gate, unitary_matrix_logarithm, compute_fidelity
    >>> t = 1.0
    >>> d = 2
    >>> U = QFT_gate(d)
    >>> H = qt.Qobj(1.0j * unitary_matrix_logarithm(U.full()))
    >>> L = 0.5 * qt.sigmaz()
    >>> gamma_search_result, AGI_search_result, search_data = find_AGI_plateau_adaptive(H, U, L, t, d)
    >>> print(f"AGI at root: {AGI_search_result}, Gamma at root: {gamma_search_result}")
    >>> print(f"Search data: {search_data}")

    References:
    ----------- 

    Changelog:
    ----------
    - 13/03/2024 : Initial commit (Jean-Gabriel Hartmann)
    - 18/03/2024 : Added the reverse adaptive search method for finding the AGI plateau root (Jean-Gabriel Hartmann)
    
    """

    # TYPE VALIDATION:
    # ----------------
    # Check if H is a Qobj
    if not isinstance(H, qt.Qobj):
        raise TypeError("Parameter H must be a Qobj")
    # Check if H is Hermitian
    if not H.isherm:
        raise TypeError("Parameter H must be Hermitian")
    # Check if U is a Qobj
    if not isinstance(U, qt.Qobj):
        raise TypeError("Parameter U must be a Qobj")
    # Check if U is a superoperator
    if not U.issuper:
        raise TypeError("Parameter U must be a superoperator")
    # Check if L is a Qobj
    if not isinstance(L, qt.Qobj):
        raise TypeError("Parameter L must be a Qobj")
    # Check if t is a float
    if not isinstance(t, float):
        raise TypeError("Parameter t must be a float")
    # Check if d is an int
    if not isinstance(d, int):
        raise TypeError("Parameter d must be an int")
    # Check if gamma_max is a float
    if not isinstance(gamma_max, float):
        raise TypeError("Parameter gamma_max must be a float")
    # Check if gamma_step is a float
    if not isinstance(gamma_step, float):
        raise TypeError("Parameter gamma_step must be a float")
    # Check if y_threshold is a float
    if not isinstance(y_threshold, float):
        raise TypeError("Parameter y_threshold must be a float")
    # Check if x_tolerance is a float
    if not isinstance(x_tolerance, float):
        raise TypeError("Parameter x_tolerance must be a float")
    # Check if options_mesolve is a qt.Options
    if not isinstance(options_mesolve, qt.Options):
        raise TypeError("Parameter options_mesolve must be a qt.Options")
    # Check if save_data is a bool
    if not isinstance(save_data, bool):
        raise TypeError("Parameter save_data must be a bool")
    
    # VALUE VALIDATION:
    # -----------------
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d")
    # Check if U is a square matrix of dimension d * d
    if U.shape[0] != U.shape[1] or U.shape[0] != d * d:
        raise ValueError("Parameter U must be a square matrix of dimension d * d")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d")
    # Check if t is positive
    if t <= 0:
        raise ValueError("Parameter t must be positive")
    # Check if d is less than 2
    if not d >= 2:
        raise ValueError("Parameter d must be greater than or equal to 2")
    # Check if gamma_max is positive
    if gamma_max <= 0:
        raise ValueError("Parameter gamma_max must be positive")
    # Check if gamma_step is positive
    if gamma_step <= 0:
        raise ValueError("Parameter gamma_step must be positive")
    # Check if gamma_step is less than gamma_max
    if gamma_step >= gamma_max:
        raise ValueError("Parameter gamma_step must be less than gamma_max")
    # Check if y_threshold is positive
    if y_threshold <= 0:
        raise ValueError("Parameter y_threshold must be positive")
    # Check if x_tolerance is positive
    if x_tolerance <= 0:
        raise ValueError("Parameter x_tolerance must be positive")
    
    # INITIALISATION:
    # ---------------
    # Initialise return variables
    g : float = gamma_max
    current_AGI : float = 0.0
    saved_data : list = []
    # Initialise local variables
    step : float = gamma_step
    plateau_AGI : float = 1 - compute_fidelity(H, U, L, t, d, g, options_mesolve)
    gammas_direction : int = -1  # Start searching towards smaller gamma values    

    # CALCULATIONS:
    # -------------
    # Reverse adaptive search for the AGI plateau root
    while step > x_tolerance:
        # Adjust gamma based on current direction and step size
        g += step * gammas_direction  
        # Calculate the AGI value for the current gamma
        current_AGI = 1 - compute_fidelity(H, U, L, t, d, g, options_mesolve)

        # Save search data if save_data is True
        if save_data:
            saved_data.append((g, current_AGI))

        # Check if current AGI crosses the threshold relative to the plateau
        # This happens if the AGI value exceeds the threshold when the search direction is negative, or if the AGI value crosses to within the threshold when the search direction is positive
        if (gammas_direction < 0 and np.abs(current_AGI - plateau_AGI) > y_threshold) or (gammas_direction > 0 and np.abs(current_AGI - plateau_AGI) < y_threshold):
            gammas_direction *= -1  # Reverse direction if threshold is crossed
            step /= 5  # Reduce step size

    # WARNINGS:
    # ---------
    # Check if the search result for gamma lies outside of the range of [0, gamma_max]
    if g < 0 or g > gamma_max:
        warnings.warn("Search result for gamma lies outside of the range of [0, gamma_max]", UserWarning)
    # Check if the search result for AGI lies outside of the range of [0, 1]
    if current_AGI < 0 or current_AGI > 1:
        warnings.warn("Search result for AGI lies outside of the range of [0, 1]", UserWarning)
    # Check if the search result for AGI does not converge to the plateau value of AGI(gamma_max) within the threshold region
    if np.abs(current_AGI - plateau_AGI) > 1.1 * y_threshold:
        warnings.warn("Search result for AGI does not converge to the plateau value of AGI(gamma_max) within the threshold region", UserWarning)

    # RETURNS:
    # --------
    return g, current_AGI, saved_data
# ----------------------------------------------------------------------------------------------------------------------------------    
def find_AGI_plateau_roots(AGIs : np.ndarray, gammas : np.ndarray, y_threshold : float = 1e-3, index : int = 0) -> tuple[int, list[tuple[float, float]]]:
    """
    Description:
    ------------
    Find the roots corresponding to all instances where the AGI curve crosses the plateau value within a certain threshold.
    This method uses the CubicSpline class from Scipy to interpolate the AGI curve and then uses the root_scalar function to find the the crossing points (roots) of the AGI curve with the upper and lower thresholds about the plateau level
    The method returns a list of tuples of the roots and the AGI values at the roots.
    The index parameter is used only to maintain the order of the roots and their corresponding AGI data during parallelisation.

    Parameters:
    -----------
    AGIs : np.ndarray
        The AGI (y-axis) values of the curve.
    gammas : np.ndarray
        The gamma (x-axis) values of the curve.
    y_threshold : float, optional (default = 1e-3)
        The threshold for the AGI plateau value.
    index : int, optional (default = 0)
        The index of the AGI data within the AGIs list.

    Returns:
    --------
    index : int
        The index of the AGI data within the AGIs list.
    roots : list[tuple[float, float]]
        The list of tuples of the roots and the corresponding AGI values at the roots.
    
    Raises:
    -------
    TypeError: Type mismatch between method parameters and input arguments. Possible reasons include:
        - AGIs is not a numpy array.
        - gammas is not a numpy array.
        - y_threshold is not a float.
        - index is not an int.

    ValueError: Inappropriate values for input arguments. Possible reasons include:
        - AGIs and gammas do not have the same length.
        - y_threshold is not positive.
        - index is negative.

    UserWarning: Possible issues with the input arguments. Possible reasons include:
        - The search result did not converge to a root within the AGI threshold and search interval of gamma values
        - The search result for gamma lies outside of the range of [0, gamma_max].
        - The search result for AGI lies outside of the range of [0, 1].
        - The search result for AGI does not converge to the plateau value of AGI(gamma_max) within the threshold region.

    Notes:
    ------
    - The returned roots list will will either be of length 1 or length 3.
    - The returned roots list is sorted in ascending order of the gamma values of the roots.    
    - The returned roots list is of length 1 if there is only one root, corresponding to the AGI curve converging monotonically to the plateau from below.
    - The returned roots list of of length 3 if there are 3 roots, corresponding to the AGI curve overshooting the plateau value once, reaching a peak and then converging monotonically to the plateau value from above.

    See Also:
    ---------
    - scipy.interpolate.CubicSpline
    - scipy.optimize.root_scalar

    Examples:
    ---------
    Assuming the following AGI and gamma values, and using the default values for x_tolerance and y_threshold:
    >>> AGIs = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
    >>> gammas = np.array([1, 2, 3, 4, 5])
    >>> roots = find_AGI_plateau_roots(AGIs, gammas)
    >>> print(roots)

    References:
    -----------
    [1] Scipy, "scipy.interpolate.CubicSpline", https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    [2] Scipy, "scipy.optimize.root_scalar", https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html

    Changelog:
    ----------
    - 13/03/2024 : Initial commit (Jean-Gabriel Hartmann)
    - 28/03/2024 : Update method to use scipy.optimize.root_scalar iteratively over intervals of gamma and return all roots (Jean-Gabriel Hartmann)
    
    """

    # TYPE VALIDATION:
    # ----------------
    # Check if AGIs is a numpy array
    if not isinstance(AGIs, np.ndarray):
        raise TypeError("Parameter AGIs must be a numpy array")
    # Check if gammas is a numpy array
    if not isinstance(gammas, np.ndarray):
        raise TypeError("Parameter gammas must be a numpy array")
    # Check if y_threshold is a float
    if not isinstance(y_threshold, float):
        raise TypeError("Parameter y_threshold must be a float")
    # Check if index is an int
    if not isinstance(index, int):
        raise TypeError("Parameter index must be an int")
    
    # VALUE VALIDATION:
    # -----------------
    # Check if AGIs and gammas have the same length
    if AGIs.shape[0] != gammas.shape[0]:
        raise ValueError("Parameters AGIs and gammas must have the same length")
    # Check if y_threshold is positive
    if y_threshold <= 0:
        raise ValueError("Parameter y_threshold must be positive")
    # Check if index is negative
    if index < 0:
        raise ValueError("Parameter index must be positive")
    
    # INITIALISATION:
    # ---------------
    # Initialise return variables
    roots : list = []
    # Initialise local variables
    AGI_cubic_spline : sp.interpolate.CubicSpline = CubicSpline(gammas, AGIs, bc_type='natural')    
    plateau_y : float = AGIs[-1]
    gammas_root_find_intervals : np.ndarray = np.geomspace(gammas[0], gammas[-1], 10)
    crossings : list = []
    search_interval : list = []
    # Initialise local functions
    def difference_upper(x):
        return AGI_cubic_spline(x) - (plateau_y + y_threshold)
    def difference_lower(x):
        return AGI_cubic_spline(x) - (plateau_y - y_threshold)

    # CALCULATIONS:
    # -------------    
    # Find roots of the functions within each interval
    for i in range(len(gammas_root_find_intervals) - 1):
        search_interval = [gammas_root_find_intervals[i], gammas_root_find_intervals[i+1]]
        # Search for crossings with upper threshold
        try:
            root_upper : RootResults = root_scalar(difference_upper, bracket = search_interval)
            if root_upper.converged:
                crossings.append(root_upper.root)
        except ValueError:
            # No root in this segment
            pass
        # Search for crossings with lower threshold
        try:
            root_lower : RootResults = root_scalar(difference_lower, bracket = search_interval)
            if root_lower.converged:
                crossings.append(root_lower.root)
        except ValueError:
            # No root in this segment
            pass
    # Sort the roots and create a list of tuples of the roots and the AGI values at the roots
    roots = [(c, AGI_cubic_spline(c)) for c in np.sort(crossings)]

    # WARNINGS:
    # ---------
    # Check if the search result did not converge to a root within the AGI threshold and search interval of gamma values
    if len(roots) == 0:
        warnings.warn("Search result did not converge to a root within the AGI threshold and search interval of gamma values", UserWarning)
    # Check if the search results for gamma lies outside of the range of [0, gamma_max]
    if any (r[0] < 0 or r[0] > gammas[-1] for r in roots):
        warnings.warn("Search result for gamma lies outside of the range of [0, gamma_max]", UserWarning)
    # Check if the search results for AGI lies outside of the range of [0, 1]
    if any (r[1] < 0 or r[1] > 1 for r in roots):
        warnings.warn("Search result for AGI lies outside of the range of [0, 1]", UserWarning)
    # Check if the search results for AGI does not converge to the plateau value of AGI(gamma_max) within the threshold region
    if any (np.abs(r[1] - plateau_y) > 1.1 * y_threshold for r in roots):
        warnings.warn("Search result for AGI does not converge to the plateau value of AGI(gamma_max) within the threshold region", UserWarning)

    # RETURN:
    # -------
    return index, roots
# ---------------------------------------------------------------------------------------------------------------------------------- 
def find_AGI_peak(AGIs: np.ndarray, gammas: np.ndarray, index: int = 0) -> tuple[int, float, float]:
    """
    Description:
    ------------
    Find the peak (maximum value) of the AGI curve using cubic spline interpolation for better accuracy.
    This method first checks if the AGI values are monotonically increasing. If they are, the peak is 
    assumed to be at the last gamma value. Otherwise, the method uses the CubicSpline class from Scipy 
    to interpolate the AGI curve and then uses the minimize_scalar function to find the maximum value 
    of the AGI curve.

    Parameters:
    -----------
    AGIs : np.ndarray
        The AGI (y-axis) values of the curve.
    gammas : np.ndarray
        The gamma (x-axis) values of the curve.
    index : int, optional (default = 0)
        The index of the AGI data within the AGIs list.

    Returns:
    --------
    index : int
        The index of the AGI data within the AGIs list.
    peak_gamma : float
        The gamma value at which the peak AGI occurs.
    peak_AGI : float
        The peak AGI value.

    Raises:
    -------
    TypeError: Type mismatch between method parameters and input arguments. Possible reasons include:
        - AGIs is not a numpy array.
        - gammas is not a numpy array.
        - index is not an int.

    ValueError: Inappropriate values for input arguments. Possible reasons include:
        - AGIs and gammas do not have the same length.
        - index is negative.

    Notes:
    ------
    - If the AGI values are monotonically increasing, the peak is assumed to be at the last gamma value.
    - If the AGI values are not monotonically increasing, the method uses cubic spline interpolation 
      to find the peak value.

    Examples:
    ---------
    Assuming the following AGI and gamma values:
    >>> AGIs = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
    >>> gammas = np.array([1, 2, 3, 4, 5])
    >>> index, peak_gamma, peak_AGI = find_AGI_peak(AGIs, gammas)
    >>> print(f"The peak AGI value is {peak_AGI} at gamma = {peak_gamma}")

    References:
    -----------
    [1] Scipy, "scipy.interpolate.CubicSpline", https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    [2] Scipy, "scipy.optimize.minimize_scalar", https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    Changelog:
    ----------
    - 17/05/2024 : Initial commit (Jean-Gabriel Hartmann)
    """
    
    # TYPE VALIDATION:
    if not isinstance(AGIs, np.ndarray):
        raise TypeError("Parameter AGIs must be a numpy array")
    if not isinstance(gammas, np.ndarray):
        raise TypeError("Parameter gammas must be a numpy array")
    if not isinstance(index, int):
        raise TypeError("Parameter index must be an int")
    
    # VALUE VALIDATION:
    if AGIs.shape[0] != gammas.shape[0]:
        raise ValueError("Parameters AGIs and gammas must have the same length")
    if index < 0:
        raise ValueError("Parameter index must be non-negative")
    
    # Check if AGIs is monotonically increasing
    if np.all(np.diff(AGIs) >= 0):
        # If the AGI values are monotonically increasing, the peak is at the last gamma value
        peak_gamma = gammas[-1]
        peak_AGI = AGIs[-1]
    else:
        # INITIALISATION:
        AGI_cubic_spline : sp.interpolate.CubicSpline = CubicSpline(gammas, AGIs, bc_type='natural') 
        
        # Local function to compute the negative of the interpolated AGI function
        def negative_AGI(x):
            return -AGI_cubic_spline(x)
        
        # CALCULATIONS:
        # Find the maximum AGI by minimizing the negative of the AGI cubic spline
        result = minimize_scalar(negative_AGI, bounds=(gammas[0], gammas[-1]), method='bounded')
        
        if result.success:
            peak_gamma = result.x
            peak_AGI = AGI_cubic_spline(peak_gamma)
        else:
            raise ValueError("Failed to find a peak of the AGI curve")
    
    # RETURN:
    return index, peak_gamma, peak_AGI
# ---------------------------------------------------------------------------------------------------------------------------------- 
def haar_random_gate(d : int) -> np.ndarray:
    """
    Description:
    ------------
    This function generates a Haar-random unitary matrix of dimension d x d.
    It is based on the algorithm described in the paper by Mezzadri (2007) [1].

    Parameters:
    -----------
    d : int
        The dimension of the unitary matrix.

    Returns:
    --------
    U : np.ndarray
        The Haar-random unitary matrix of dimension d x d.

    Raises:
    -------
    TypeError
        If the input parameter is not an integer.
    ValueError
        If the input parameter is less than 2.

    See Also:
    ---------
    qutip.random_objects.rand_unitary_haar

    Notes:
    ------
    The algorithm uses the QR decomposition of a matrix whose entries are complex standard normal random variables.
    The diagonal R matrix is normalised so that the product Q.R'.Q is distributed with Haar measure.

    References:
    -----------
    [1] F. Mezzadri, "How to generate random matrices from the classical compact groups", Notices of the AMS, 54, 592-604 (2007).

    Examples:
    ---------
    >>> haar_measure(2)
    array([[ 0.70710678+0.j,  0.70710678+0.j],
           [-0.70710678+0.j,  0.70710678+0.j]])

    Changelog:
    ----------
    - 07/05/2024 : Initial commit (Jean-Gabriel Hartmann)
    """

    # Type Verification
    if not isinstance(d, int):
        raise TypeError("The input parameter must be an integer.")
    
    # Value Verification
    if d < 2:
        raise ValueError("The input parameter must be greater than or equal to 2.")

    # Initialise return variables
    U : np.ndarray = np.zeros((d, d), dtype = complex)

    # Initialise local variables
    Z : np.ndarray = np.zeros((d, d), dtype = complex)
    Q : np.ndarray = np.zeros((d, d), dtype = complex)
    R : np.ndarray = np.zeros((d, d), dtype = complex)
    R_diag : np.ndarray = np.zeros(d, dtype = complex)
    R_norm : np.ndarray = np.zeros(d, dtype = complex)

    # Calculations
    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = np.linalg.qr(Z)
    R_diag = np.diagonal(R)
    R_norm = R_diag / np.abs(R_diag)
    U = np.multiply(Q, R_norm, Q)

    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def DFT_gate(d : int) -> qt.Qobj:
    """Create a DFT matrix of size d x d.
    
    Parameters:
    -----------
    d : int
        The dimension of the DFT matrix.

    Returns:
    --------
    U : qt.Qobj
        A d x d DFT matrix.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2.
    """

    # Check that d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check that d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    
    # Initialise return variables
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # Initialise local variables
    u : np.ndarray = np.zeros((d, d), dtype = np.complex128)

    # Create the DFT matrix by the FFT of the identity matrix
    u = np.fft.fft(np.eye(d)) / np.sqrt(d)

    # Convert the matrix to a Qobj
    U = qt.Qobj(u)

    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def QFT_gate(d : int) -> qt.Qobj:
    """
    Returns the QFT gate for a d-dimensional system.

    Parameters:
    -----------
    d : int
        Dimension of the system.

    Returns:
    --------
    U : qt.Qobj
        QFT gate for a d-dimensional system.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2.
    """

    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")

    # Initialise return variables
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # Initialise local variables
    idx : np.ndarray = np.arange(d, dtype = np.complex128)
    u : np.ndarray = np.zeros((d, d), dtype = np.complex128)    
    
    # Create the QFT matrix
    u : np.ndarray = idx[:, np.newaxis] * idx[np.newaxis, :]
    
    u *= 2j * np.pi / d
    
    np.exp(u, out = u)
    
    u /= np.sqrt(d)
    
    # Convert the matrix to a Qobj
    U = qt.Qobj(u)
    
    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def SHIFT_gate(d : int) -> qt.Qobj :
    """
    Returns the SHIFT gate for a d-dimensional system.

    Parameters:
    -----------
    d : int
        Dimension of the system.
    
    Returns:
    --------
    U : qt.Qobj
        SHIFT gate for a d-dimensional system.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2.
    """

    # Check if d is an integer and is greater than 1.
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")

    # Initialise the SHIFT matrix
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))
    
    # Define the Indentity matrix
    identity : np.ndarray = np.eye(d, dtype = np.complex128)

    # Define the INC gate matrix
    u : np.ndarray = np.roll(identity, shift = -1, axis = 0)
    
    # Convert the matrix to a Qobj
    U = qt.Qobj(u)
    
    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def CLOCK_gate(d : int) -> qt.Qobj :
    """
    Returns the CLOCK gate for a d-dimensional system.

    Parameters:
    -----------
    d : int
        Dimension of the system.
    
    Returns:
    --------
    U : qt.Qobj
        CLOCK gate for a d-dimensional system.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2.
    """

    # Check if d is an integer and is greater than 1.
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")

    # Initialise the CLOCK matrix
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))
    
    # Define an array of the dth roots of unity
    diag : np.ndarray = np.array([np.exp(1.0j * 2 * np.pi * i / d) for i in range(d)])

    # Define the diagonal matrix with the dth roots of unity on the diagonal
    u : np.ndarray = np.diag(diag)
    
    # Convert the matrix to a Qobj
    U = qt.Qobj(u)
    
    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def interpolated_INC(d : int, eta : float = 1.0, shift : int = -1, imag : bool = False) -> qt.Qobj :
    """
    Generate a unitary matrix that interpolates between the identity
    and the INC/ SHIFT gate based on the parameter eta.

    Parameters:
    -----------
    d : int
        Dimension of the system.

    eta : float
        Interpolation parameter in the range [0, 1].
        Default value is 1.0.
    
    shift : int
        Shift parameter for the INC/ SHIFT gate.
        Default value is -1.

    imag : bool
        If True, the INC/ SHIFT gate is imaginary.
        Default value is False.

    Returns:
    --------
    U : qt.Qobj
        Interpolated SHIFT gate.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2, or
        If eta is not a float, or
        If eta is not in the range [0, 1].
        If shift is not an integer, or
        If shift is not equal to +1 or -1, or
        If imag is not a bool.
    """

    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if eta is a float
    if not isinstance(eta, float):
        raise ValueError("Parameter eta must be a float.")
    # Check if eta is in the range [0, 1]
    if eta < 0 or eta > 1:
        raise ValueError("Parameter eta must be in the range [0, 1].")
    # Check if shift is an integer
    if not isinstance(shift, int):
        raise ValueError("Parameter shift must be an integer.")
    # Check if shift is equal to +1 or -1
    if shift != +1 and shift != -1:
        raise ValueError("Parameter shift must be equal to +1 or -1.")
    # Check if imag is a bool
    if not isinstance(imag, bool):
        raise ValueError("Parameter imag must be a bool.")
    
    # Initialise the unitary matrix
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # Define the identity matrix
    identity : np.ndarray = np.eye(d, dtype = np.complex128)

    if imag:

        identity = 1.0j * identity

        if shift == +1:

            identity[d-1][d-1] = -1.0j
        else:
            identity[0][0] = -1.0j

    # Define the INC gate matrix
    X_d : np.ndarray = np.roll(identity, shift = shift, axis = 0)

    # Find the Hermitian generator of X_d (logarithm of the unitary matrix)
    H_X_d : np.ndarray = 1.0j * sp.linalg.logm(X_d) # type: ignore

    # Interpolate between the zero matrix and H_X_d
    H_interpolated : np.ndarray = eta * H_X_d  # because the zero matrix contributes nothing

    # Construct the unitary by exponentiating the interpolated Hermitian
    u : np.ndarray = sp.linalg.expm(-1.0j * H_interpolated)

    # Convert the matrix to a Qobj
    U = qt.Qobj(u)
    
    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def interpolated_CLK(d : int, eta : float = 1.0) -> qt.Qobj :
    """
    Generate a unitary matrix that interpolates between the identity and the CLOCK gate based on the parameter eta.

    Parameters:
    -----------
    d : int
        The dimension of the qudit system.

    eta : float
        The interpolation parameter in the range [0, 1].

    Returns:
    --------
    U : qt.Qobj
        Interpolated CLOCK gate.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2, or
        If eta is not a float, or
        If eta is not in the range [0, 1].
    """

    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if eta is a float
    if not isinstance(eta, float):
        raise ValueError("Parameter eta must be a float.")
    # Check if eta is in the range [0, 1]
    if eta < 0 or eta > 1:
        raise ValueError("Parameter eta must be in the range [0, 1].")
    
    # Initialise the unitary matrix
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # Define an array of the dth roots of unity
    diag : np.ndarray = np.array([np.exp(1.0j * 2 * np.pi * i / d) for i in range(d)])

    # Define the diagonal matrix with the dth roots of unity on the diagonal
    Z_d : np.ndarray = np.diag(diag)

    # Find the Hermitian generator of Z_d (logarithm of the unitary matrix)
    H_Z_d : np.ndarray = 1.0j * sp.linalg.logm(Z_d) # type: ignore

    # Interpolate between the zero matrix and H_Z_d
    H_interpolated : np.ndarray = eta * H_Z_d  # because the zero matrix contributes nothing

    # Construct the unitary by exponentiating the interpolated Hermitian
    u : np.ndarray = sp.linalg.expm(-1.0j * H_interpolated)

    # Convert the matrix to a Qobj
    U = qt.Qobj(u)

    return U   
# ----------------------------------------------------------------------------------------------------------------------------------
def generalised_SWAP(d : int, swap_state_1 : int, swap_state_2 : int) -> np.ndarray:
    """
    Generate a matrix representation of a custom gate that swaps two specified states.

    Parameters:
    d (int): The dimension of the system (number of energy levels).
    swap_state_1 (int): The first state to be swapped (integer from 0 to d-1).
    swap_state_2 (int): The second state to be swapped (integer from 0 to d-1).

    Returns:
    numpy.ndarray: A d x d matrix representing the custom gate.
    """
    if d < 2:
        raise ValueError("Dimension d must be 2 or greater")
    
    if not(0 <= swap_state_1 < d) or not(0 <= swap_state_2 < d):
        raise ValueError("Swap states must be between 0 and d-1")
    
    if swap_state_1 == swap_state_2:
        raise ValueError("Swap states must be different")

    # Start with a d x d identity matrix since most states are unchanged
    gate = np.eye(d, dtype = np.complex128)

    # The gate swaps the |swap_state_1⟩ and |swap_state_2⟩ states, so we modify those entries
    gate[swap_state_1, swap_state_1] = 0
    gate[swap_state_2, swap_state_2] = 0
    gate[swap_state_1, swap_state_2] = 1
    gate[swap_state_2, swap_state_1] = 1

    return gate
# ----------------------------------------------------------------------------------------------------------------------------------
def generate_swap_state_pairs(d):
    """
    Generate all possible unique pairs of states for swapping in a d-dimensional system.

    Parameters:
    d (int): The dimension of the system (number of energy levels).

    Returns:
    list[tuple[int, int]]: A list of tuples, where each tuple contains two integers representing a pair of states.
    """
    if d < 2:
        raise ValueError("Dimension d must be 2 or greater for swaps to be meaningful.")

    # Generate a list of all possible states
    states = list(range(d))

    # Use combinations to generate all unique pairs of states
    pairs = list(combinations(states, 2))

    return pairs
# ----------------------------------------------------------------------------------------------------------------------------------
def fixed_gates(d : int, gate : str) -> qt.Qobj:
    """
    Returns a fixed gate for a d-dimensional system.

    Parameters:
    -----------
    d : int
        Dimension of the system.

    gate : str
        The gate to be returned. Must be one of the following:
        'DFT', 'QFT', 'SHIFT', 'CLOCK', 'IDENTITY'.
    
    Returns:
    --------
    U : qt.Qobj
        Fixed gate for a d-dimensional system.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2, or
        If gate is not a string, or
        If gate is not one of the following: 'DFT', 'QFT', 'SHIFT', 'CLOCK', 'IDENTITY'
    """

    # Check if d is an integer and is greater than 1.
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if gate is a string
    if not isinstance(gate, str):
        raise ValueError("Parameter gate must be a string.")
    # Check if gate is one of the following
    if gate != 'DFT' and gate != 'QFT' and gate != 'SHIFT' and gate != 'CLOCK' and gate != 'IDENTITY':
        raise ValueError("Parameter gate must be one of the following: 'DFT', 'QFT', 'SHIFT', 'CLOCK', 'IDENTITY'.")
    
    # Initialise the fixed gate
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # Define the fixed gate
    if gate == 'DFT':
        U = DFT_gate(d)
    elif gate == 'QFT':
        U = QFT_gate(d)
    elif gate == 'SHIFT':
        U = SHIFT_gate(d)
    elif gate == 'CLOCK':
        U = CLOCK_gate(d)
    elif gate == 'IDENTITY':
        U = qt.Qobj(np.eye(d, dtype = np.complex128))

    return U
# ----------------------------------------------------------------------------------------------------------------------------------
def von_neumann_entropy(rho : qt.Qobj, method : str = 'eig') -> float :
    """
    Calculate the von Neumann entropy of a density matrix.

    Parameters:
    -----------
    rho : qt.Qobj
        The density matrix.
    method : str, optional (default = 'eig')
        The method to use to calculate the entropy.
        Options are : 'eig' (eigenvalues) or 'mat' (matrix logarithm).

    Returns:
    --------
    S : float
        The von Neumann entropy.

    Raises:
    -------
    TypeError
        If the parameter rho is not a qutip Qobj, or
        If the parameter method is not a string.
    ValueError
        If the parameter rho is not a square matrix, or
        If the parameter rho is not unit trace, or
        If the parameter method is not a valid option.
    """

    # Check if rho is a qutip Qobj
    if not isinstance(rho, qt.Qobj):
        raise TypeError('Parameter rho must be of type qutip.Qobj')
    
    # Check if method is a string
    if not isinstance(method, str):
        raise TypeError('Parameter method must be of type str')
    
    # Check if rho is a square matrix
    if rho.shape[0] != rho.shape[1]:
        raise ValueError('Parameter rho must be a square matrix')
    
    # Check if rho is unit trace
    if not np.isclose(rho.tr(), 1.0):
        raise ValueError('Parameter rho must be unit trace')
    
    # Check if method is a valid option
    if method not in ['eig', 'mat']:
        raise ValueError('Parameter method must be either "eig" or "mat"')
    
    # Initialise return variables
    S : float = 0.0

    # Initialise local variables
    eigs : np.ndarray = np.array([])

    if method == 'eig':
        eigs = rho.eigenenergies()
        S = -sum([e * np.log2(e) for e in eigs if e > 1e-15])
        
    elif method == 'mat':
        S = np.trace(rho * sp.linalg.logm(rho))

    return S
# ----------------------------------------------------------------------------------------------------------------------------------
def von_neumann_entropy_wrapper(gate_idx, state_idx, gamma_idx, rho):
    entropy = von_neumann_entropy(rho)
    return gate_idx, state_idx, gamma_idx, entropy
# ----------------------------------------------------------------------------------------------------------------------------------
def compute_propagator(H : qt.Qobj, L : qt.Qobj, t : float, g : float, d : int, options : qt.Options) -> qt.Qobj :
    """
    Compute the propagator for the given Hamiltonian and Lindbladian.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.

    L : qt.Qobj
        The Lindbladian of the system.

    t : float
        The time at which to compute the propagator.

    g : float
        The strength of the control field.

    d : int
        The dimension of the system.

    options : qt.Options
        Options for the propagator.

    Returns:
    --------
    propagator : List[qt.Qobj]
        The propagator for the given Hamiltonian and Lindbladian.

    Raises:
    -------
    ValueError
        If H is not a qt.Qobj, or
        If H is not a square matrix of dimension d, or
        If H is not Hermitian, or
        If L is not a qt.Qobj, or
        If L is not a square matrix of dimension d, or
        If t is not a float, or
        If t is less than 0, or
        If g is not a float, or
        If g is less than 0, or
        If d is not an integer, or
        If d is less than 2, or
        If options is not a qt.Options object.
    """

    # Check if H is a qt.Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a qt.Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is Hermitian
    if not H.isherm:
        # Property isherm is not always accurate for large numerical values, so we use the following check as a backup
        if not np.allclose(H.full(), H.dag().full()):
            raise ValueError("Parameter H must be Hermitian.")
    # Check if L is a qt.Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a qt.Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than or equal to 0.")
    # Check if g is a float
    if not isinstance(g, float):
        raise ValueError("Parameter g must be a float.")
    # Check if g is greater than 0
    if g < 0:
        raise ValueError("Parameter g must be greater than or equal to 0.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if options is a qt.Options object
    if not isinstance(options, qt.Options):
        raise ValueError("Parameter options must be a qt.Options object.")
    
    # Initialise return variables
    propagator : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # Initialise local variables
    c_ops : list[qt.Qobj] = [qt.Qobj(np.zeros((d, d), dtype = np.complex128))]

    # Calculate the collapse operators
    c_ops = [np.sqrt(g) * L]

    # Calculate the propagator
    propagator = qt.propagator(H, t, c_op_list = c_ops, options = options)

    return propagator
# ----------------------------------------------------------------------------------------------------------------------------------
def propagate_density_matrix(H : qt.Qobj, rho_initial : qt.Qobj, L : qt.Qobj, t : float, d : int, g : float, options_mesolve : qt.Options) -> qt.Qobj:
    """
    Compute the coherences of a density matrix after evolution by a Hamiltonian.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.

    rho_initial : qt.Qobj
        The initial density matrix of the system.

    L : qt.Qobj
        The collapse operator of the system.

    t : float
        The time point at which to compute the coherences.

    d : int
        The dimension of the system.

    g : float
        The noise strength of the system.

    options_mesolve : qt.Options
        Options for the mesolve function.

    Returns:
    --------
    rho_final : qt.Qobj
        The final density matrix of the system.

    Raises:
    -------
    ValueError
        If H is not a Qobj, or
        If H is not a square matrix of dimension d, or
        If H is not Hermitian, or
        If rho_initial is not a Qobj, or
        If rho_initial is not a square matrix of dimension d, or
        If rho_initial is not unit trace, or
        If L is not a Qobj, or
        If L is not a square matrix of dimension d, or
        If t is not a float, or
        If t is less than 0, or
        If d is not an integer, or
        If d is less than 2, or
        If g is not a float, or
        If g is less than 0, or        
        If options_mesolve is not a qt.Options object.
    """

    # Check if H is a Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is Hermitian
    if not H.isherm:
        # Property isherm is not always accurate for large numerical values, so we use the following check as a backup
        if not np.allclose(H.full(), H.dag().full()):
            raise ValueError("Parameter H must be Hermitian.")
    # Check if rho_initial is a Qobj
    if not isinstance(rho_initial, qt.Qobj):
        raise ValueError("Parameter rho_initial must be a Qobj.")
    # Check if rho_initial is a square matrix of dimension d
    if rho_initial.shape[0] != rho_initial.shape[1] or rho_initial.shape[0] != d:
        raise ValueError("Parameter rho_initial must be a square matrix of dimension d.")
    # Check if rho_initial is unit trace
    if np.trace(rho_initial) != 1:
        # Property trace is not always accurate to numerical precision, so we use the following check as a backup
        if not np.allclose(np.trace(rho_initial), 1):
            raise ValueError("Parameter rho_initial must be unit trace.")
    # Check if L is a Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than 0.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if g is a float
    if not isinstance(g, float):
        raise ValueError("Parameter g must be a float.")
    # Check if g is greater than 0
    if g < 0:
        raise ValueError("Parameter g must be greater than 0.")
    # Check if options_mesolve is a qt.Options object
    if not isinstance(options_mesolve, qt.Options):
        raise ValueError("Parameter options_mesolve must be a qt.Options object.")
    
    # Initialise return variables
    rho_final : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128), type = 'dm')
    
    # Initialise local variables
    n_coherences : int = int(d * (d - 1) / 2)
    coherences : np.ndarray = np.zeros(n_coherences, dtype = float)
    propagators : list[qt.Qobj] = []
    propagator_list : list[qt.Qobj] = []
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))    

    # This is because of the way the qutip.mesolve function works. 
    # If times is a single float, the function only returns a single propagator for that time, so we need to make it into a list.
    # Furthermore, the propagator function calls the QuTiP mesolve function to perform time evolution, and requires the times list to have 0 as the first element.
    # Therefore, we add a zero to the list of times.
    # times = np.concatenate(([0], times))

    # Compute the propagators
    propagators = compute_propagator(H, L, t, g, d, options_mesolve) # type:ignore

    # This is because of the way the qutip.propagator function works.
    # If the list of times is exactly of length 2, e.g. [0, t], the function only returns the final propagator as a Qobj.
    # # If the list of times is greater than length 2, the function returns a list of Qobj propagators; then ignore the first element in the list of propagators corresponding to time = 0.
    if isinstance(propagators, qt.Qobj):
        propagator_list = [propagators]
    else:
        propagator_list = propagators[1:]

    U = propagator_list[0]   

    # Compute the coherences by evolving the density matrix
    rho_final = U(rho_initial) #type:ignore

    return rho_final
# ----------------------------------------------------------------------------------------------------------------------------------
def propagate_density_matrix_wrapper(gate_idx, state_idx, gamma_idx, hamiltonian, state, L, times, d, gamma, options_mesolve):
        density_matrix = propagate_density_matrix(hamiltonian, state, L, times, d, gamma, options_mesolve)
        return gate_idx, state_idx, gamma_idx, density_matrix
# ----------------------------------------------------------------------------------------------------------------------------------
def compute_coherences(H : qt.Qobj, rho_initial : qt.Qobj, L : qt.Qobj, t : float, d : int, g : float, options_mesolve : qt.Options) -> float:
    """
    Compute the coherences of a density matrix after evolution by a Hamiltonian.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.

    rho_initial : qt.Qobj
        The initial density matrix of the system.

    L : qt.Qobj
        The collapse operator of the system.

    t : float
        The time point at which to compute the coherences.

    d : int
        The dimension of the system.

    g : float
        The noise strength of the system.

    options_mesolve : qt.Options
        Options for the mesolve function.

    Returns:
    --------
    average_coherences : float
        The mean coherences of the superoperator.

    Raises:
    -------
    ValueError
        If H is not a Qobj, or
        If H is not a square matrix of dimension d, or
        If H is not Hermitian, or
        If rho_initial is not a Qobj, or
        If rho_initial is not a square matrix of dimension d, or
        If rho_initial is not unit trace, or
        If L is not a Qobj, or
        If L is not a square matrix of dimension d, or
        If t is not a float, or
        If t is less than 0, or
        If d is not an integer, or
        If d is less than 2, or
        If g is not a float, or
        If g is less than 0, or        
        If options_mesolve is not a qt.Options object.
    """

    # Check if H is a Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is Hermitian
    if not H.isherm:
        # Property isherm is not always accurate for large numerical values, so we use the following check as a backup
        if not np.allclose(H.full(), H.dag().full()):
            raise ValueError("Parameter H must be Hermitian.")
    # Check if rho_initial is a Qobj
    if not isinstance(rho_initial, qt.Qobj):
        raise ValueError("Parameter rho_initial must be a Qobj.")
    # Check if rho_initial is a square matrix of dimension d
    if rho_initial.shape[0] != rho_initial.shape[1] or rho_initial.shape[0] != d:
        raise ValueError("Parameter rho_initial must be a square matrix of dimension d.")
    # Check if rho_initial is unit trace
    if np.trace(rho_initial) != 1:
        raise ValueError("Parameter rho_initial must be unit trace.")
    # Check if L is a Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than 0.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if g is a float
    if not isinstance(g, float):
        raise ValueError("Parameter g must be a float.")
    # Check if g is greater than 0
    if g < 0:
        raise ValueError("Parameter g must be greater than 0.")
    # Check if options_mesolve is a qt.Options object
    if not isinstance(options_mesolve, qt.Options):
        raise ValueError("Parameter options_mesolve must be a qt.Options object.")
    
    # Initialise return variables
    average_coherences : float = 0.0
    
    # Initialise local variables
    n_coherences : int = int(d * (d - 1) / 2)
    coherences : np.ndarray = np.zeros(n_coherences, dtype = float)
    propagators : list[qt.Qobj] = []
    propagator_list : list[qt.Qobj] = []
    U : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))
    rho_final : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))

    # This is because of the way the qutip.mesolve function works. 
    # If times is a single float, the function only returns a single propagator for that time, so we need to make it into a list.
    # Furthermore, the propagator function calls the QuTiP mesolve function to perform time evolution, and requires the times list to have 0 as the first element.
    # Therefore, we add a zero to the list of times.
    # times = np.concatenate(([0], times))

    # Compute the propagators
    propagators = compute_propagator(H, L, t, g, d, options_mesolve) # type:ignore

    # This is because of the way the qutip.propagator function works.
    # If the list of times is exactly of length 2, e.g. [0, t], the function only returns the final propagator as a Qobj.
    # # If the list of times is greater than length 2, the function returns a list of Qobj propagators; then ignore the first element in the list of propagators corresponding to time = 0.
    if isinstance(propagators, qt.Qobj):
        propagator_list = [propagators]
    else:
        propagator_list = propagators[1:]

    U = propagator_list[0]   

    # Compute the coherences by evolving the density matrix
    rho_final = U(rho_initial) #type:ignore

    # Get the coherences from the final density matrix
    coherences = coherences_density_matrix(rho_final)

    # Compute the average coherences
    average_coherences = np.mean(coherences) # type:ignore

    return average_coherences
# ----------------------------------------------------------------------------------------------------------------------------------
def compute_fidelity(H : qt.Qobj, super_gate : qt.Qobj, L : qt.Qobj, t : float, d : int, g : float, options_mesolve : qt.Qobj) -> float:
    """
    Compute the fidelity of a superoperator.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.
    super_gate : qt.Qobj
        The superoperator to be compared to.
    L : qt.Qobj
        The collapse operator of the system.
    times : float
        The time point at which to compute the propagator.
    d : int
        The dimension of the system.
    g : float
        The noise strength of the system.
    options_mesolve : qt.Options
        Options for the mesolve function.

    Returns:
    --------
    fidelity : float
        The fidelity of the superoperator.

    Raises:
    -------
    ValueError
        If H is not a Qobj, or
        If H is not a square matrix of dimension d, or
        If H is not Hermitian, or
        If super_gate is not a Qobj, or
        If super_gate is not a square matrix of dimension d*d, or
        If super_gate is not a superoperator, or
        If L is not a Qobj, or
        If L is not a square matrix of dimension d, or
        If t is not a float, or
        If t is less than 0, or
        If d is not an integer, or
        If d is less than 2, or
        If g is not a float, or
        If g is less than 0, or        
        If options_mesolve is not a qt.Options object.
    """

    # Check if H is a Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is Hermitian
    if not H.isherm:
        # Property isherm is not always accurate for large numerical values, so we use the following check as a backup
        if not np.allclose(H.full(), H.dag().full(), atol = 1e-2, rtol = 1e-2):
            raise ValueError("Parameter H must be Hermitian.")
    # Check if super_gate is a Qobj
    if not isinstance(super_gate, qt.Qobj):
        raise ValueError("Parameter super_gate must be a Qobj.")
    # Check if super_gate is a square matrix of dimension d*d
    if super_gate.shape[0] != super_gate.shape[1] or super_gate.shape[0] != d*d:
        raise ValueError("Parameter super_gate must be a square matrix of dimension d*d.")
    # Check if super_gate is a superoperator
    if not super_gate.issuper:
        raise ValueError("Parameter super_gate must be a superoperator.")
    # Check if L is a Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than 0.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if g is a float
    if not isinstance(g, float):
        raise ValueError("Parameter g must be a float.")
    # Check if g is greater than 0
    if g < 0:
        raise ValueError("Parameter g must be greater than 0.")
    # Check if options_mesolve is a qt.Options object
    if not isinstance(options_mesolve, qt.Options):
        raise ValueError("Parameter options_mesolve must be a qt.Options object.")
    
    # Initialise return variable
    average_gate_fidelity : float = 0.0

    # Initialise temporary variables
    propagator : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))
    super_propagator : qt.Qobj = qt.Qobj(np.zeros((d*d, d*d), dtype = np.complex128))
    process_fidelity_val : float = 0.0

    # Compute the propagator
    propagator = compute_propagator(H, L, t, g, d, options_mesolve)
    
    # Convert the propagator to a superoperator
    super_propagator = qt.to_super(propagator)

    # Compute the process fidelity
    process_fidelity_val = process_fidelity(super_propagator, super_gate)

    # Compute the average gate fidelity
    average_gate_fidelity = (d * np.real(process_fidelity_val) + 1) / (d + 1)
    
    return average_gate_fidelity
# ----------------------------------------------------------------------------------------------------------------------------------
def compute_fidelity_wrapper(gamma_idx, gate_idx, hamiltonian, gate, L, times, d, gamma, options):
    """
    Description:
    ------------
    Wrapper function to compute the fidelity of a superoperator.

    Parameters:
    -----------
    gamma_idx : int
        Index of the gamma value.
    gate_idx : int
        Index of the gate.
    hamiltonian : qt.Qobj
        The Hamiltonian of the system.
    gate : qt.Qobj
        The superoperator to be compared to.
    L : qt.Qobj
        The collapse operator of the system.
    times : float
        The time point at which to compute the propagator.
    d : int
        The dimension of the system.
    gamma : float
        The noise strength of the system.
    options : qt.Options
        Options for the mesolve function.

    Returns:
    --------

    Raises:
    -------

    Notes:
    ------

    Examples:
    ---------

    References:
    -----------

    Changes:
    --------
    """
    fidelities = compute_fidelity(hamiltonian, gate, L, times, d, gamma, options)
    return gamma_idx, gate_idx, fidelities
# ----------------------------------------------------------------------------------------------------------------------------------
def compute_fidelitys(H : qt.Qobj, super_gate : qt.Qobj, L : qt.Qobj, t : float, d : int, g : float, options_mesolve : qt.Qobj) -> float:
    """
    Compute the fidelity of a superoperator.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.
    super_gate : qt.Qobj
        The superoperator to be compared to.
    L : qt.Qobj
        The collapse operator of the system.
    times : float
        The time point at which to compute the propagator.
    d : int
        The dimension of the system.
    g : float
        The noise strength of the system.
    options_mesolve : qt.Options
        Options for the mesolve function.

    Returns:
    --------
    fidelity : float
        The fidelity of the superoperator.

    Raises:
    -------
    ValueError
        If H is not a Qobj, or
        If H is not a square matrix of dimension d, or
        If H is not Hermitian, or
        If super_gate is not a Qobj, or
        If super_gate is not a square matrix of dimension d*d, or
        If super_gate is not a superoperator, or
        If L is not a Qobj, or
        If L is not a square matrix of dimension d, or
        If t is not a float, or
        If t is less than 0, or
        If d is not an integer, or
        If d is less than 2, or
        If g is not a float, or
        If g is less than 0, or        
        If options_mesolve is not a qt.Options object.
    """

    # Check if H is a Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is Hermitian
    if not H.isherm:
        # Property isherm is not always accurate for large numerical values, so we use the following check as a backup
        if not np.allclose(H.full(), H.dag().full()):
            raise ValueError("Parameter H must be Hermitian.")
    # Check if super_gate is a Qobj
    if not isinstance(super_gate, qt.Qobj):
        raise ValueError("Parameter super_gate must be a Qobj.")
    # Check if super_gate is a square matrix of dimension d*d
    if super_gate.shape[0] != super_gate.shape[1] or super_gate.shape[0] != d*d:
        raise ValueError("Parameter super_gate must be a square matrix of dimension d*d.")
    # Check if super_gate is a superoperator
    if not super_gate.issuper:
        raise ValueError("Parameter super_gate must be a superoperator.")
    # Check if L is a Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than 0.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if g is a float
    if not isinstance(g, float):
        raise ValueError("Parameter g must be a float.")
    # Check if g is greater than 0
    if g < 0:
        raise ValueError("Parameter g must be greater than 0.")
    # Check if options_mesolve is a qt.Options object
    if not isinstance(options_mesolve, qt.Options):
        raise ValueError("Parameter options_mesolve must be a qt.Options object.")
        
    # Initialise return variables
    average_gate_fidelity : float = 0.0

    # Initialise local variables
    propagator : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))
    super_propagator : qt.Qobj = qt.Qobj(np.zeros((d*d, d*d), dtype = np.complex128), type = 'super')
    process_fidelity_val : float = 0.0

    # Compute the propagator
    propagator = compute_propagator(H, L, t, g, d, options_mesolve)
    
    # Convert the propagator to a superoperator
    super_propagator = qt.to_super(propagator)

    # Compute the process fidelity
    process_fidelity_val = process_fidelity(super_propagator, super_gate)

    # Compute the average gate fidelity
    average_gate_fidelity = (d * np.real(process_fidelity_val) + 1) / (d + 1)
    
    return average_gate_fidelity
# ----------------------------------------------------------------------------------------------------------------------------------
def generate_single_gate_and_hamiltonian(method_function, dim, *args, **kwargs):
    """
    Generate a single unitary gate and its corresponding Hamiltonian.
    If the optional argument is_super is True, generate a single superoperator and its corresponding Hamiltonian, or
    If the optional argument is_hermitian is True, generate a single Hamiltonian and its corresponding unitary gate.

    Parameters:
    -----------
    method_function : callable
        The function to create the gate matrix.
    dim : int
        The dimension of the system (number of energy levels / states).
    *args : list
        Additional arguments for the method_function.
    **kwargs : dict
        Additional keyword arguments for the method_function and the gate generation process.

    Returns:
    --------
    gate : qt.Qobj
        The unitary gate.
    hamiltonian : qt.Qobj
        The Hamiltonian.

    Raises:
    -------
    TypeError
        If method_function is not callable, or
        If dim is not an integer.
    ValueError
        If dim is less than 2.

    Examples:
    ---------
    >>> U, H = generate_single_gate_and_hamiltonian(qt.rand_unitary, 2)
    """

    # Check if method_function is callable
    if not callable(method_function):
        raise TypeError("Parameter method_function must be callable.")
    
    # Check if dim is an integer
    if not isinstance(dim, int):
        raise TypeError("Parameter dim must be an integer.")
    
    # Check if dim is greater than 1
    if dim < 2:
        raise ValueError("Parameter dim must be greater than 1.")

    # Initialise return variables
    U : qt.Qobj = qt.Qobj(np.zeros((dim, dim), dtype = np.complex128))
    H : qt.Qobj = qt.Qobj(np.zeros((dim, dim), dtype = np.complex128))

    # Initialise local variables
    is_hermitian : bool = kwargs.pop('is_hermitian', False)
    is_super : bool = kwargs.pop('is_super', False)
    gate_matrix : np.ndarray = np.zeros((dim, dim), dtype = np.complex128)

    # Generate the gate matrix using the provided method_function and additional arguments and keyword arguments
    gate_matrix = method_function(dim, *args, **kwargs)
    U = qt.Qobj(gate_matrix)

    # Some generation methods return a Hermitian (Hamiltonian) matrix directly instead of a unitary gate, for example qt.rand_herm()
    if is_hermitian:
        # If the matrix is a Hamiltonian, convert it directly to a gate by matrix exponentiation
        H = U
        U = qt.Qobj(sp.linalg.expm(-1.0j * H.full()))
    else:
        # Otherwise, calculate the Hamiltonian matrix from the generated unitary gate matrix, e.g.
        # H = qt.Qobj(1.0j * sp.linalg.logm(U.full()))
        # However, not that the Scipy logm function is not accurate for certain special cases, so we use the following method instead
        H = qt.Qobj(1.0j * unitary_matrix_logarithm(U.full()))
        
    if is_super:
        U = to_super(U)

    return U, H
# ----------------------------------------------------------------------------------------------------------------------------------
def generate_gates_and_hamiltonians(n_gates : int, dim : int, method: str, n_jobs : int = 10, batch_size : int = 1000, is_super : bool = False, etas : list[float] = []) -> tuple[list[Qobj], list[Qobj]]:
    """
    Generate a list of unitary gates and their corresponding Hamiltonians using the specified gate generation method.

    Parameters:
    -----------
    n_gates : int
        The number of gates to generate.
    dim : int
        The dimension of the system (number of energy levels / states).
    method : str
        The method to use to generate the gates.
        Options are defined in variable method_map.
    n_jobs : int (optional, default = 10)
        The number of parallel jobs to run.
    batch_size : int (optional, default = 1000)
        The number of gates to generate in each batch of parallel jobs.
    is_super : bool (optional, default = False)
        If True, generate superoperators instead of unitary gates.
    etas : list[float] (optional, default = [])
        The list of eta values for the interpolated gates.
    
    Returns:
    --------
    gates : list[qt.Qobj]
        The list of unitary gates.
    hamiltonians : list[qt.Qobj]
        The list of Hamiltonians.

    Raises:
    -------
    TypeError
        If n_gates is not an integer, or
        If dim is not an integer, or
        If method is not a string, or
        If n_jobs is not an integer, or
        If batch_size is not an integer, or
        If is_super is not a boolean, or
        If etas is not a list of floats.
    ValueError
        If n_gates is less than 1, or
        If dim is less than 2, or
        If method is not a valid option, or
        If n_jobs is less than 1, or
        If batch_size is less than 1, or
        If etas is not a list of floats.

    Examples:
    ---------
    >>> gates, hamiltonians = generate_gates_and_hamiltonians(10, 2, 'haar')
    """
    
    # Check if n_gates is an integer
    if not isinstance(n_gates, int):
        raise TypeError("Parameter n_gates must be an integer.")
    # Check if n_gates is greater than 0
    if n_gates < 1:
        raise ValueError("Parameter n_gates must be greater than 0.")
    # Check if dim is an integer
    if not isinstance(dim, int):
        raise TypeError("Parameter dim must be an integer.")
    # Check if dim is greater than 1
    if dim < 2:
        raise ValueError("Parameter dim must be greater than 1.")
    # Check if method is a string
    if not isinstance(method, str):
        raise TypeError("Parameter method must be a string.")
    # Check if n_jobs is an integer
    if not isinstance(n_jobs, int):
        raise TypeError("Parameter n_jobs must be an integer.")
    # Check if n_jobs is greater than 0
    if n_jobs < 1:
        raise ValueError("Parameter n_jobs must be greater than 0.")
    # Check if batch_size is an integer
    if not isinstance(batch_size, int):
        raise TypeError("Parameter batch_size must be an integer.")
    # Check if batch_size is greater than 0
    if batch_size < 1:
        raise ValueError("Parameter batch_size must be greater than 0.")
    # Check if is_super is a boolean
    if not isinstance(is_super, bool):
        raise TypeError("Parameter is_super must be a boolean.")
    # Check if etas is a list of floats
    if not all(isinstance(eta, float) for eta in etas):
        raise TypeError("Parameter etas must be a list of floats.")

    # Initialise return variables
    gates : list[qt.Qobj] = []
    hamiltonians : list[qt.Qobj] = []

    # Initialise local variables
    ce = Circular()
    d = dim # Alias for readability
    method_map = {
        "circular": (ce.gen_cue, {}),
        "haar": (qt.rand_unitary_haar, {}),
        "random": (qt.rand_unitary, {'density' : 1.0}),
        "hermitian": (qt.rand_herm, {'is_hermitian': True}),
        "cirq_random_unitary": (random_unitary, {}),
        "cirq_random_special_unitary": (random_special_unitary, {}),
        "generalised_SWAP": (generalised_SWAP, {}),
        'INC': (),
        'DFT' : (DFT_gate, {}),
        'QFT' : (QFT_gate, {}),
        'SHIFT' : (SHIFT_gate, {}),
        'CLOCK' : (CLOCK_gate, {}),
        "interpolated_INC": (interpolated_INC, {}),
        "interpolated_CLK": (interpolated_CLK, {}),
        "fixed" : (fixed_gates, {}),
        "haar_measure" : (haar_random_gate, {})        
    }
    # Check for unsupported methods
    if method not in method_map:
        raise ValueError(f"Unsupported method: {method}")
    # Get the method function and its special kwargs
    method_function, special_kwargs = method_map[method]
    # General arguments that are passed to the single gate generation function
    general_args = [d]
    general_kwargs = {'is_super': is_super}

    # Prepare task arguments
    if method in ['fixed']:
        gates = ['IDENTITY', 'QFT', 'SHIFT']
        tasks = (general_args + [gate] for gate in gates)
    elif method in ['interpolated_INC']:
        tasks = (general_args + [eta] for eta in etas)
    elif method in ['interpolated_CLK']:
        tasks = (general_args + [eta] for eta in etas)
    elif method in ['generalised_SWAP']:
        # Generate all possible pairs of states for swapping
        all_state_pairs = generate_swap_state_pairs(dim)  # or use your function to generate pairs
        tasks = (general_args + list(state_pair) for state_pair in all_state_pairs)
        n_gates = len(all_state_pairs)
        n_jobs = min(10, n_gates)
        batch_size = int(n_gates / n_jobs)
    else:
        tasks = (general_args for _ in range(n_gates))
    # Prepare kwargs (merging general and special ones)
    merged_kwargs = {**general_kwargs, **special_kwargs}

    # Parallel computation
    results = Parallel(n_jobs = n_jobs, batch_size = batch_size, verbose = 10)(delayed(generate_single_gate_and_hamiltonian)(method_function, *task_args, **merged_kwargs) for task_args in tasks) # type: ignore

    # Unzipping the results into two lists
    gates, hamiltonians = zip(*results)

    return list(gates), list(hamiltonians)
# ----------------------------------------------------------------------------------------------------------------------------------
def first_order_AGI(d : int, L : qt.Qobj) -> float :
    """
    Calculate the first order AGI correction term for a given Collapse Operator L
    
    Parameters:
    -----------
    d : int
        The dimension of the system.
    L : qt.Qobj
        The collapse operator.

    Returns:
    --------
    AGI_1 : float
        The first order AGI correction.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2, or
        If L is not a Qobj, or
        If L is not a square matrix of dimension d.
    """

    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if L is a Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != d or L.shape[1] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")

    # Initialise return variables
    AGI_1 : float = 0.0

    # Initialise local variables
    Tr_L : float = 0.0

    # Compute the traces
    Tr_L = (np.abs(np.trace(L))**2 - d * np.trace(L.dag() * L))

    # Compute the first order AGI correction
    AGI_1 = Tr_L / (d * (d + 1))

    return AGI_1
# ----------------------------------------------------------------------------------------------------------------------------------
def nested_commutator_trace_oper(H : qt.Qobj, L : qt.Qobj, d : int, s : int) -> tuple[qt.Qobj, float] :
    """
    Calculate the nested commutator trace for a given Hamiltonian H and collapse operator L to order s.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian of the system.
    L : qt.Qobj
        The collapse operator of the system.
    d : int
        The dimension of the system.
    s : int
        The order of the nested commutator trace.

    Returns:
    --------
    nested_comm : qt.Qobj
        The nested commutator.
    nested_trace : float
        The nested commutator trace.

    Raises:
    -------
    ValueError
        If H is not a qt.Qobj, or
        If H is not a square matrix of dimension d, or
        If H is not of type 'oper', or
        If H is not Hermitian, or
        If L is not a qt.Qobj, or
        If L is not a square matrix of dimension d, or
        If L is not of type 'oper', or
        If d is not an integer, or
        If d is less than 2, or
        If s is not an integer, or
        If s is less than 0.
    """

    # Check if H is a qt.Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a qt.Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] or H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is of type 'oper'
    if H.type != 'oper':
        raise ValueError("Parameter H must be of type 'oper'.")
    # Check if H is Hermitian
    if not H.isherm:
        raise ValueError("Parameter H must be Hermitian.")
    # Check if L is a qt.Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a qt.Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] or L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if L is of type 'oper'
    if L.type != 'oper':
        raise ValueError("Parameter L must be of type 'oper'.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if s is an integer
    if not isinstance(s, int):
        raise ValueError("Parameter s must be an integer.")
    # Check if s is greater than or equal to 0
    if s < 0:
        raise ValueError("Parameter s must be greater than or equal to 0.")
    
    # Initialise return variables
    nested_comm : qt.Qobj = qt.Qobj(np.zeros((d, d), dtype = np.complex128))
    nested_trace : float = 0.0

    # Initialise local variables
    curr_trace : float = 0.0
    term0 : float = 0.0
    term1 : float = 0.0
    term2 : float = 0.0
    term3 : float = 0.0
    term4 : float = 0.0
    term5 : float = 0.0
    term6 : float = 0.0
    term7 : float = 0.0
    term8 : float = 0.0
    term9 : float = 0.0
    
    # Convert L, L_dag and H from Qobj to np.array
    l_dag : np.ndarray = L.dag().full()
    l : np.ndarray = L.full()
    h : np.ndarray = H.full()

    for n in range(s + 1):
        for k in range(s - n + 1):
            for j in range(n + 1):
                
                term0 = ((-1)**(n + k + j) * factorial(s)) / (factorial(k) * factorial(j) * factorial(n - j) * factorial(s - n - k))
                term1 = np.trace(l_dag @ fractional_matrix_power(h, s - n - k) @ l_dag @ fractional_matrix_power(h, n - j))
                term2 = np.trace(l @ fractional_matrix_power(h, k) @ l @ fractional_matrix_power(h, j))
                term3 = np.trace(l @ l_dag @ fractional_matrix_power(h, s - n - k) @ l @ l_dag @ fractional_matrix_power(h, n - j))
                term4 = np.trace(fractional_matrix_power(h, k + j))
                term5 = np.trace(l_dag @ fractional_matrix_power(h, s - n - k) @ l @ l_dag @ fractional_matrix_power(h, n - j))
                term6 = np.trace(l @ fractional_matrix_power(h, k + j))
                term7 = np.trace(l_dag @ fractional_matrix_power(h, n - j) @ l @ l_dag @ fractional_matrix_power(h, s - n - k))
                term8 = np.trace(fractional_matrix_power(h, s - k - j) @ l @ l_dag)
                term9 = np.trace(l @ l_dag @ fractional_matrix_power(h, k + j))

                curr_trace = term0 * (term1 * term2 + 0.5 * term3 * term4 - np.real(term5 * term6) - np.real(term7 * term6) + 0.5 * np.real(term8 * term9))
                
                nested_trace += (-1.0j)**s * curr_trace # type:ignore

    return nested_comm, nested_trace
# ----------------------------------------------------------------------------------------------------------------------------------   
def  nested_commutator_trace_super(S : qt.Qobj, L : qt.Qobj, C : qt.Qobj, d : int) -> tuple[qt.Qobj, float] :
    """
    Calculate the nested commutator and trace for a given Hamiltonian operator S, collapse operator L and nested commutator C.

    Parameters:
    -----------
    S : qt.Qobj
        The Hamiltonian operator.
    L : qt.Qobj
        The collapse operator.
    C : qt.Qobj
        The nested commutator.
    d : int
        The dimension of the system.

    Returns:
    --------
    nested_comm : qt.Qobj
        The nested commutator.
    nested_trace : float
        The nested commutator trace.

    Raises:
    -------
    ValueError
        If S is not a qt.Qobj, or
        If S is not a square matrix, or
        If S is not a superoperator of dimension d*d, or
        If L is not a qt.Qobj, or
        If L is not a square matrix, or
        If L is not a superoperator of dimension d*d, or
        If C is not a qt.Qobj, or
        If C is not a square matrix, or
        If C is not a superoperator of dimension d*d, or
        If d is not an integer, or
        If d is less than 2.
    """

    # Check if S is a qt.Qobj
    if not isinstance(S, qt.Qobj):
        raise ValueError("Parameter S must be a qt.Qobj.")
    # Check if S is a square matrix
    if S.shape[0] != S.shape[1]:
        raise ValueError("Parameter S must be a square matrix.")
    # Check if S is a superoperator and of dimension d*d
    if S.type != 'super' or S.shape[0] != d*d:
        raise ValueError("Parameter S must be a superoperator of dimension d*d.")
    # Check if L is a qt.Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a qt.Qobj.")
    # Check if L is a square matrix
    if L.shape[0] != L.shape[1]:
        raise ValueError("Parameter L must be a square matrix.")
    # Check if L is a superoperator and of dimension d*d
    if L.type != 'super' or L.shape[0] != d*d:
        raise ValueError("Parameter L must be a superoperator of dimension d*d.")
    # Check if C is a qt.Qobj
    if not isinstance(C, qt.Qobj):
        raise ValueError("Parameter C must be a qt.Qobj.")
    # Check if C is a square matrix
    if C.shape[0] != C.shape[1]:
        raise ValueError("Parameter C must be a square matrix.")
    # Check if C is a superoperator and of dimension d*d
    if C.type != 'super' or C.shape[0] != d*d:
        raise ValueError("Parameter C must be a superoperator of dimension d*d.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    
    # Initialise return variables
    nested_comm : qt.Qobj = qt.Qobj(np.zeros(d * d, dtype = np.complex128), type = 'super')
    nested_trace : float = 0.0

    # Calculate the nested commutator
    nested_comm = qt.operators.commutator(S, qt.operators.commutator(S, C)) # type:ignore

    # Calculate the nested commutator trace
    nested_trace = np.trace(L * nested_comm)

    return nested_comm, nested_trace
# ----------------------------------------------------------------------------------------------------------------------------------
def nested_commutator_loop_summation(S_super : qt.Qobj, L_super : qt.Qobj, t : float, s_max : int, error_threshold : float) -> tuple[float, int] :
    """
    DEPRECATED
    Calculate the nested commutator sum using a while loop. The loop terminates when the error between two successive iterations is less than the error_threshold or when the number of iterations reaches s_max.

    Parameters:
    -----------
    S_super : qt.Qobj
        The superoperator corresponding to the Hamiltonian.
    L_super : qt.Qobj
        The superoperator corresponding to the collapse operator.
    t : float
        The time parameter.
    s_max : int
        The maximum number of iterations.
    error_threshold : float
        The error threshold for the loop.
    
    Returns:
    --------
    comm_sum : float
        The nested commutator sum.

    Raises:
    -------
    ValueError
        If the error threshold is not positive, or
        if the maximum number of iterations is not positive, or
        If t is not positive, or
        if the superoperators are not square matrices of the same dimension.
    """

    # Check that the error threshold is positive
    if error_threshold <= 0:
        raise ValueError("The error threshold must be positive.")
    
    # Check that the maximum number of iterations is positive
    if s_max <= 0:
        raise ValueError("The maximum number of iterations must be positive.")
    
    # Check that t is positive
    if t <= 0:
        raise ValueError("The time parameter must be positive.")
    
    # Check that the superoperators are square matrices of the same dimension
    if S_super.shape[0] != S_super.shape[1] or L_super.shape[0] != L_super.shape[1] or S_super.shape[0] != L_super.shape[0]:
        raise ValueError("The superoperators must be square matrices of the same dimension.")
    
    # Initialize the previous commutator
    prev_comm : qt.Qobj = L_super

    # Initialize the previous iteration 
    prev_term : float = 0.5 * np.trace(L_super * prev_comm)
    
    # Initialize the current commutator
    curr_comm : qt.Qobj = qt.Qobj(np.zeros(S_super.shape, dtype=np.complex128))

    # Initialize the current commutator term
    curr_term : float = 0.0

    # Initialize the commutator sum
    comm_sum : float = 0.0

    # Initialize the error
    error : float = np.inf

    # Initialize the iteration counter
    # The counter starts at 2 because the zeroth-order term is simply L_super and calculated outside this function
    curr_s : int = 2    

    # Loop until the error is less than the error threshold or the maximum number of iterations is reached
    # while error > error_threshold and curr_s <= s_max:
    while curr_s <= s_max:        
        
        # Calculate the current iteration
        curr_comm = qt.operators.commutator(S_super, qt.operators.commutator(S_super, prev_comm))

        curr_term = (-t)**curr_s * np.trace(L_super * curr_comm) / sp.special.factorial(curr_s + 2)

        # Update the commutator sum
        comm_sum += curr_term

        # Calculate the error
        error = np.abs(curr_term - prev_term)     

        # Diagnostics
        print(f"curr_s: {curr_s}")
        print(f"error: {error}")

        if error < error_threshold:
            return comm_sum, curr_s   

        # Update the previous iteration
        prev_term = curr_term

        # Increment the iteration counter
        curr_s += 2

    final_s = curr_s - 2



    # Print the final error and iteration count
    print(f"Final error: {error}")
    print(f"Final iteration count: {final_s}")

    return comm_sum, final_s
# ----------------------------------------------------------------------------------------------------------------------------------
def nested_commutator_sum(S : qt.Qobj, L : qt.Qobj, d : int, t : float, s_max : int, error_threshold : float = 0.0, super : bool = False) -> tuple[float, int] :
    """
    Calculate the nested commutator sum using a while loop. The loop terminates when the error between two successive iterations is less than the error_threshold or when the number of iterations reaches s_max.

    Parameters:
    -----------
    S : qt.Qobj
        The operator corresponding to the Hamiltonian.
    L : qt.Qobj
        The operator corresponding to the collapse operator.
    d : int
        The dimension of the system.
    t : float
        The time parameter.
    s_max : int
        The maximum number of iterations.
    error_threshold : float
        The error threshold for the loop.
        The default value is 0.0, which means the loop will run for s_max iterations.
    super : bool
        A flag to indicate if the operators are superoperators.
        The default value is False, which means the operators are operators.
    
    Returns:
    --------
    comm_sum : float
        The nested commutator sum.
    s_final : int
        The final number of iterations.

    Raises:
    -------
    ValueError
        If S is not a qt.Qobj, or
        If S is not a square matrix, or
        If S is of type 'super' and of dimension d*d, or
        If S is of type 'oper' and of dimension d, or
        If L is not a qt.Qobj, or
        If L is not a square matrix, or
        If L is of type 'super' and of dimension d*d, or
        If L is of type 'oper' and of dimension d, or
        If S and L are not of the same type, or
        If d is not an integer, or
        If d is less than 2, or
        If t is not a float, or
        If t is less than 0, or
        If s_max is not an integer, or
        If s_max is less than 0, or
        If error_threshold is not a float, or
        If error_threshold is less than 0, or
        If super is not a bool, or
        If super is True and S or L are not superoperators, or
        If super is False and S or L are not operators.
    """

    # Check if S is a qt.Qobj
    if not isinstance(S, qt.Qobj):
        raise ValueError("Parameter S_super must be a qt.Qobj.")
    # Check if S is a square matrix
    if S.shape[0] != S.shape[1]:
        raise ValueError("Parameter S_super must be a square matrix.")
    # Check if S is of type 'super' and of dimension d*d
    if S.type == 'super' and S.shape[0] != d*d:
        raise ValueError("Parameter S is of type 'super' but not of dimension d*d.")
    # Check if S is of type 'oper' and of dimension d
    if S.type == 'oper' and S.shape[0] != d:
        raise ValueError("Parameter S is of type 'oper' but not of dimension d.")
    # Check if L is a qt.Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L_super must be a qt.Qobj.")
    # Check if L is a square matrix
    if L.shape[0] != L.shape[1] and L.shape[0] != d*d:
        raise ValueError("Parameter L must be a square matrix.")
    # Check if L is of type 'super' and of dimension d*d
    if L.type == 'super' and L.shape[0] != d*d:
        raise ValueError("Parameter L is of type 'super' but not of dimension d*d.")
    # Check if L is of type 'oper' and of dimension d
    if L.type == 'oper' and L.shape[0] != d:
        raise ValueError("Parameter L is of type 'oper' but not of dimension d.")
    # Check that both S and L are of the same type
    if S.type != L.type:
        raise ValueError("Parameters S and L must be of the same type.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than 0.")
    # Check if s_max is an integer
    if not isinstance(s_max, int):
        raise ValueError("Parameter s_max must be an integer.")
    # Check if s_max is greater than or equal to 0
    if s_max < 0:
        raise ValueError("Parameter s_max must be greater than or equal to 0.")
    # Check if error_threshold is a float
    if not isinstance(error_threshold, float):
        raise ValueError("Parameter error_threshold must be a float.")
    # Check if error_threshold is greater than 0
    if error_threshold < 0:
        raise ValueError("Parameter error_threshold must be greater than 0.")
    # Check if super is a bool
    if not isinstance(super, bool):
        raise ValueError("Parameter super must be a bool.")
    # Check if super is True and S or L are not superoperators
    if super and (S.type != 'super' or L.type != 'super'):
        raise ValueError("Parameters S and L must be superoperators.")
    # Check if super is False and S or L are not operators
    if not super and (S.type != 'oper' or L.type != 'oper'):
        raise ValueError("Parameters S and L must be operators.")

    # Initialise return variables
    comm_sum : float = 0.0
    s_final : int = 0

    # Initialise local variables
    if super:
        prev_comm : qt.Qobj = qt.Qobj(np.zeros(d * d, dtype = np.complex128), type = 'super')
        curr_comm : qt.Qobj = qt.Qobj(np.zeros(d * d, dtype = np.complex128), type = 'super')        
    else:
        prev_comm : qt.Qobj = qt.Qobj(np.zeros(d, dtype = np.complex128), type = 'oper')
        curr_comm : qt.Qobj = qt.Qobj(np.zeros(d, dtype = np.complex128), type = 'oper')   

    prev_trace : float = 0.0
    curr_trace : float = 0.0

    prev_term : float = 0.0    
    curr_term : float = 0.0

    error : float = np.inf
    curr_s : int = 0
    
    # Define the previous commutator
    if super:
        prev_comm, prev_trace = L, np.trace(L*L)
    else:
        prev_comm, prev_trace = nested_commutator_trace_oper(S, L, d, 0)

    # Define the zeroth-order in s term
    prev_term = 0.5 * prev_trace

    comm_sum = prev_term

    # If s_max is 0 or 1, the commutator sum is simply the zeroth-order term
    if s_max == 0 or s_max == 1:
        comm_sum = prev_term
        s_final = s_max
        return comm_sum, s_final

    # Initialize the iteration counter for the while loop for s_max greater than 1
    # The counter starts at 2 because the zeroth-order term is simply L_super and returned above
    curr_s = 2    

    # If the error threshold is 0, set it to -1 so that the while loop runs for s_max iterations
    if error_threshold == 0.0: error_threshold = -1.0

    # Loop until the error is less than the error threshold or the maximum number of iterations is reached
    # while error > error_threshold and curr_s <= s_max:
    while curr_s <= s_max:
        
        # Calculate the current iteration
        if super:
            curr_comm, curr_trace = nested_commutator_trace_super(S, L, prev_comm, d)
        else:
            curr_comm, curr_trace = nested_commutator_trace_oper(S, L, d, curr_s)

        curr_term = (-t)**curr_s * curr_trace / sp.special.factorial(curr_s + 2)

        # Update the commutator sum
        comm_sum += curr_term

        # Calculate the error
        error = np.abs(curr_term - prev_term)

        if error <= error_threshold:
            # print(f"Termination due to ERROR = {error} at S_FINAL = {curr_s} and d = {d}")
            return comm_sum, curr_s

        # print(f"{curr_s} : {curr_term / (d * (d + 1))}")

        # Update the previous iteration
        prev_comm = curr_comm
        prev_term = curr_term

        # Increment the iteration counter
        curr_s += 2

    # Set the final iteration count because curr_s is incremented by 2 at the end of the while loop
    s_final = curr_s - 2

    # Print the final error and iteration count
    # print(f"Termination due to s_final = {s_final} at error = {error}")

    return comm_sum, s_final
# ----------------------------------------------------------------------------------------------------------------------------------
def second_order_AGI(H : qt.Qobj, L : qt.Qobj, d : int, t : float, s_max : int, error_threshold : float = 0.0, super : bool = False) -> tuple[float, int] :
    """ 
    Calculate the L-S-commutator-product-term and the number of iterations for a given L and H.

    Parameters:
    -----------
    H : qt.Qobj
        The Hamiltonian.
    L : qt.Qobj
        The collapse operator.
    d : int
        The dimension of the system.
    t : float
        The time parameter.
    s_max : int
        The maximum number of iterations.
    error_threshold : float
        The error threshold for the loop.
        The default value is 0.0, which means the loop will run for s_max iterations.
    super : bool
        Whether the superoperator or the matrix form of the L-S-commutator-sum-term is calculated.
        The default value is False, which means the operator form is calculated.

    Returns:
    --------
    AGI_2 : float
        The L-S-commutator-sum-term.
    s_final : int
        The number of iterations.

    Raises:
    -------
    TypeError: Type mismatch between method parameters and input arguments. Possible reasons include:
        - H is not a qt.Qobj.
        - L is not a qt.Qobj.
        - d is not an integer.
        - t is not a float.
        - s_max is not an integer.
        - error_threshold is not a float.
        - super is not a bool.

    ValueError: Inappropriate values for input arguments. Possible reasons include:
        - H is not a square matrix of dimension d.
        - H is not Hermitian.
        - L is not a square matrix of dimension d.
        - d is less than 2.
        - t is less than 0.
        - s_max is less than 0.
        - error_threshold is less than 0.
    """

    # Check if H is a qt.Qobj
    if not isinstance(H, qt.Qobj):
        raise ValueError("Parameter H must be a qt.Qobj.")
    # Check if H is a square matrix of dimension d
    if H.shape[0] != H.shape[1] and H.shape[0] != d:
        raise ValueError("Parameter H must be a square matrix of dimension d.")
    # Check if H is Hermitian
    if not H.isherm:
        # Property isherm is not always accurate for large numerical values, so we use the following check as a backup
        if not np.allclose(H.full(), H.dag().full()):
            raise ValueError("Parameter H must be Hermitian.")
    # Check if L is a qt.Qobj
    if not isinstance(L, qt.Qobj):
        raise ValueError("Parameter L must be a qt.Qobj.")
    # Check if L is a square matrix of dimension d
    if L.shape[0] != L.shape[1] and L.shape[0] != d:
        raise ValueError("Parameter L must be a square matrix of dimension d.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if t is a float
    if not isinstance(t, float):
        raise ValueError("Parameter t must be a float.")
    # Check if t is greater than 0
    if t < 0:
        raise ValueError("Parameter t must be greater than 0.")
    # Check if s_max is an integer
    if not isinstance(s_max, int):
        raise ValueError("Parameter s_max must be an integer.")
    # Check if s_max is greater than or equal to 0
    if s_max < 0:
        raise ValueError("Parameter s_max must be greater than or equal to 0.")
    # Check if error_threshold is a float
    if not isinstance(error_threshold, float):
        raise ValueError("Parameter error_threshold must be a float.")
    # Check if error_threshold is greater than 0
    if error_threshold < 0:
        raise ValueError("Parameter error_threshold must be greater than 0.")
    # Check if super is a bool
    if not isinstance(super, bool):
        raise ValueError("Parameter super must be a bool.")

    # Initialise return variables
    AGI_2 : float = 0.0
    s_final : int = 0

    # Initialise local variables
    comm_sum : float = 0.0
    S : qt.Qobj = H

    if super:
        # Calculate the superoperators
        S = -1.0j * (qt.spre(H) - qt.spost(H))
        L = qt.sprepost(L, L) - qt.spre(L**2) / 2 - qt.spost(L**2) / 2

    comm_sum, s_final = nested_commutator_sum(S, L, d, t, s_max, error_threshold = error_threshold, super = super)

    # Calculate the AGI_2 correction term
    AGI_2 = comm_sum / (d * (d + 1))

    return AGI_2, s_final
# ----------------------------------------------------------------------------------------------------------------------------------
def compute_all_iterated_commutators(S: Qobj, L: Qobj, N: int) -> list:
    """
    Compute and cache all iterated commutators [S^n, L] for n in [0, N].
    """
    commutators = [L]  # [S^0, L] = L
    for n in range(1, N + 1):
        new_commutator = qt.operators.commutator(S, commutators[-1])
        commutators.append(new_commutator)
    return commutators

def compute_AGIs_up_to_mth_order(d : int, M_order : int, N : int, t : float, H : qt.Qobj, L : qt.Qobj) -> list[float]:
    """
    Description:
    ------------
    Calculate the normalized trace of M^{(m)}(t) using precomputed iterated commutators for each order m in [1, M_order].

    Parameters:
    -----------
    d : int
        Dimension of the qudit.
    M_order : int
        Order of the M operator.
    N : int
        Maximum order of the iterated commutators.
    t : float
        Time parameter.
    S : qt.Qobj
        System Hamiltonian.
    L : qt.Qobj
        Lindblad operator.

    Returns:
    --------
    list[float]
        List of normalized traces of M^{(m)}(t) for each order m in [1, M_order].

    Raises:
    -------

    Notes:
    ------

    References:
    -----------

    Examples:
    ---------

    Changelog:
    ----------

    """

    S = -1.0j * (qt.spre(H) - qt.spost(H))
    L = qt.sprepost(L, L) - qt.spre(L**2) / 2 - qt.spost(L**2) / 2

    AGI_mth_order = []

    # Precompute all iterated commutators up to order N
    commutators = compute_all_iterated_commutators(S, L, N)

    for m in range(1, M_order + 1):
        # Initialize the total sum for M^(m)(t) to zero
        M_commutator = Qobj(np.zeros((d*d, d*d)), dims=[S.dims[0], S.dims[1]])

        # Generate all combinations of indices from 0 to N for each of the m terms
        for n in np.ndindex((N+1,)*m):
            if sum(n)<(N+1):
                product_term = Qobj(np.eye(d*d), dims=S.dims)  # Start with the identity matrix
                for i in range(m):
                    ni = n[i]
                    # Calculate the inner sum from i to m
                    inner_sum = sum(n[j] + 1 for j in range(i, m))
                    
                    # Update the product term with the precomputed iterated commutator
                    product_term *= ((-t)**ni * commutators[ni]) / (sp.special.factorial(ni) * inner_sum)

                # Add to the total matrix M^(m)(t)
                M_commutator += product_term


        # Compute the trace and normalize by d*(d+1)
        trace_M = np.real(M_commutator.tr())
        normalized_trace = -trace_M / (d * (d + 1))

        AGI_mth_order.append(normalized_trace)

    return AGI_mth_order
# ----------------------------------------------------------------------------------------------------------------------------------
def log_linear_interpolation(d1 : int, y1 : float, d2 : int, y2 : float, d : int) -> float :
    """
    Interpolates on a log-linear plot given two data points and a value for d.

    Parameters:
    -----------
    d1 : int
        The first x-value.
    y1 : float
        The y-value corresponding to d1.
    d2 : int
        The second x-value.
    y2 : float
        The y-value corresponding to d2.
    d : int
        The x-value for which to interpolate the y-value.
    
    Returns:
    --------
    y : float
        The interpolated y-value.

    Raises:
    -------
    ValueError
        If d1 is not an integer, or
        If d2 is not an integer, or
        If d is not an integer, or
        If d1 is greater than d2, or
        If d is less than d1, or
        If d is greater than d2.
    """

    # Check if d1 is an integer
    if not isinstance(d1, int):
        raise ValueError("Parameter d1 must be an integer.")
    # Check if d2 is an integer
    if not isinstance(d2, int):
        raise ValueError("Parameter d2 must be an integer.")
    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d1 is greater than d2
    if d1 > d2:
        raise ValueError("Parameter d1 must be less than or equal to d2.")
    # Check if d is less than d1
    if d < d1:
        raise ValueError("Parameter d must be greater than or equal to d1.")
    # Check if d is greater than d2
    if d > d2:
        raise ValueError("Parameter d must be less than or equal to d2.")
    
    # Initialise local variables
    m1 : float = 0.0
    m2 : float = 0.0
    slope : float = 0.0
    intercept : float = 0.0
    m : float = 0.0

    # Intialise return value
    y : float = 0.0

    # Calculate the logarithms of y1 and y2
    m1 = np.log10(y1)
    m2 = np.log10(y2)

    # Calculate the slope of the line in the d-m plane
    slope = (m2 - m1) / (d2 - d1)

    # Calculate the y-intercept of the line
    intercept = m1 - slope * d1

    # Calculate the interpolated value of m
    m = slope * d + intercept

    # Convert back to the original y value
    y = 10 ** m

    return y
# ----------------------------------------------------------------------------------------------------------------------------------
def gammas_in_linear_regime(d : int, n : int) -> np.ndarray :
    """
    Returns an array of gamma values in the linear regime of the AGI.
    The gamma values are spaced logarithmically, and the range of gamma values is determined by the dimension of the system and empirical data.

    Parameters:
    -----------
    d : int
        The dimension of the system.

    n : int
        The number of points to return.

    Returns:
    --------
    gammas : np.array
        The array of gamma values in the linear regime.

    Raises:
    -------
    ValueError
        If d is not an integer, or
        If d is less than 2, or
        If n is not an integer, or
        If n is less than 2.
    """

    # Check if d is an integer
    if not isinstance(d, int):
        raise ValueError("Parameter d must be an integer.")
    # Check if d is greater than 1
    if d < 2:
        raise ValueError("Parameter d must be greater than 1.")
    # Check if n is an integer
    if not isinstance(n, int):
        raise ValueError("Parameter n must be an integer.")
    # Check if n is greater than 1
    if n < 2:
        raise ValueError("Parameter n must be greater than 1.")
    
    # Initialise return variables
    gammas : np.ndarray = np.zeros(n, dtype = np.float64)
    
    # Initialise local variables    
    d1 : int = 0
    d2 : int = 0
    y1 : float = 0.0
    y2 : float = 0.0    
    y_min : float = 0.0
    y_max : float = 0.0
    y : float = 0.0
    gamma_min : float = 0.0
    gamma_max : float = 0.0

    # Set the interpolation parameters
    d1 = 2
    d2 = 10
    y1 = 5e-3
    y2 = 5e-4
    y_min = 1.0
    y_max = 2.75

    # Determine the interpolated value based on the dimension of the system using log-linear interpolation
    y = log_linear_interpolation(d1, y1, d2, y2, d)

    # Scale the range of gamma values based on the interpolation parameters
    gamma_min = y * y_min
    gamma_max = y * y_max

    # Generate the array of gamma values logarithmically
    gammas = np.geomspace(gamma_min, gamma_max, n)

    return gammas
# ----------------------------------------------------------------------------------------------------------------------------------
def get_list_dimensions(lst : list) -> list[int]:
    """
    Returns the dimensions of a multidimensional list.

    Parameters:
    -----------
    lst : list[]
        A multidimensional list.

    Returns:
    --------
    dimensions : list[int]
        A list of the dimensions of the input list.

    Raises:
    -------
    TypeError
        If the input is not a list

    Examples:
    ---------
    >>> get_dimensions([1, 2, 3, 4])
    [4]
    >>> get_dimensions([[1, 2, 3], [4, 5, 6]])
    [2, 3]
    """

    # Initialise return variables
    dimensions = []
        
    if not isinstance(lst, list) or not lst:
        return dimensions
    
    dimensions = [len(lst)] + get_list_dimensions(lst[0])

    return dimensions
# ----------------------------------------------------------------------------------------------------------------------------------
import qutip.operators as op
import qutip.superoperator as sop
import qutip.metrics as qtm
import time, os, sys, datetime
from typing import List, Union

class ScriptDiag():
    
    def __init__(self, diag_flag = True):
        
        self.diag_flag = diag_flag
        
        self.start = time.time()
        
        pass
    
    
    

def array2func(arrayH,times):
    
    def f(t,*args):
        if t<times[-1] : 
            return arrayH[np.argmax(times>t)-1]
        else:
            return 0       
        
    return f

    
def pulses_to_H(pulses, controls):

    if len(pulses)!=len(controls):
        print("Pulses and controls size mismatch")
        raise 
        
    if callable(pulses[0]):
        return [ [controls[k],pulses[k]] for k in range(len(controls))]
    else:
        return [ [controls[k],pulses[k].func] for k in range(len(controls))]
    
    
def overlap2(a,b):
        
    return np.trace(a.dag()*b)

def process_fidelity(oper, target=None) -> float:
    """
    Returns the process fidelity of a quantum channel to the target
    channel, or to the identity channel if no target is given.
    The process fidelity between two channels is defined as the state
    fidelity between their normalized Choi matrices.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`/list
        A unitary operator, or a superoperator in supermatrix, Choi or
        chi-matrix form, or a list of Kraus operators
    target : :class:`qutip.Qobj`/list
        A unitary operator, or a superoperator in supermatrix, Choi or
        chi-matrix form, or a list of Kraus operators

    Returns
    -------
    fid : float
        Process fidelity between oper and target, or between oper and identity.

    Notes
    -----
    Since Qutip 5.0, this function computes the process fidelity as defined
    for example in: A. Gilchrist, N.K. Langford, M.A. Nielsen,
    Phys. Rev. A 71, 062310 (2005). Previously, it computed a function
    that is now implemented in
    :func:`control.fidcomp.FidCompUnitary.get_fidelity`.
    The definition of state fidelity that the process fidelity is based on
    is the one from R. Jozsa, Journal of Modern Optics, 41:12, 2315 (1994).
    It is the square of the one implemented in
    :func:`qutip.core.metrics.fidelity` which follows Nielsen & Chuang,
    "Quantum Computation and Quantum Information"

    """
    if target is None:
        return _process_fidelity_to_id(oper)

    dims_out, dims_in = _hilbert_space_dims(oper)
    if dims_out != dims_in:
        raise NotImplementedError('Process fidelity only implemented for '
                                  'dimension-preserving operators.')
    dims_out_target, dims_in_target = _hilbert_space_dims(target)
    if [dims_out, dims_in] != [dims_out_target, dims_in_target]:
        raise TypeError('Dimensions of oper and target do not match')    

    if not isinstance(target, list) and target.type == 'oper':
        # interpret target as unitary.
        if isinstance(oper, list):  # oper is a list of Kraus operators
            return _process_fidelity_to_id([k * target.dag() for k in oper])
        elif oper.type == 'oper':
            return _process_fidelity_to_id(oper*target.dag())
        elif oper.type == 'super':
            oper_super = qt.to_super(oper)
            target_dag_super = qt.to_super(target.dag())
            return _process_fidelity_to_id(oper_super * target_dag_super)
    else:  # target is a list of Kraus operators or a superoperator
        if not isinstance(oper, list) and oper.type == 'oper':
            return process_fidelity(target, oper)  # reverse order

        oper_choi = _kraus_or_qobj_to_choi(oper)
        target_choi = _kraus_or_qobj_to_choi(target)

        d = np.prod(dims_in)
        return (fidelity(oper_choi, target_choi)/d)**2
    
def fidelity(A, B):
    """
    Calculates the fidelity (pseudo-metric) between two density matrices.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    Examples
    --------
    >>> x = fock_dm(5,3)
    >>> y = coherent_dm(5,1)
    >>> np.testing.assert_almost_equal(fidelity(x,y), 0.24104350624628332)
    """
    if A.isket or A.isbra:
        if B.isket or B.isbra:
            # The fidelity for pure states reduces to the modulus of their
            # inner product.
            return np.abs(A.overlap(B))
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        sqrtmA = qt.ket2dm(A)
    else:
        if B.isket or B.isbra:
            # Swap the order so that we can take a more numerically
            # stable square root of B.
            return fidelity(B, A)
        # If we made it here, both A and B are operators, so
        # we have to take the sqrtm of one of them.   
        sqrtmA = A.sqrtm() # However, sqrtm() returns a Qobj with superrep 'super', see note below       

    if sqrtmA.dims != B.dims:
        raise TypeError('Density matrices do not have same dimensions.')

    # JG HARTMANN: 14-03-2024
    # Add a warning filter to suppress the warning about the multiplication of superoperators of different super representations
    # This warning is caused by the multiplication of the superoperators sqrtmA and B, which have different super representations
    # The warning is not relevant to the user, as the result of the multiplication is a valid superoperator
    # Parameters A and B both have superoperor representations 'choi'
    # However, the sqrtm() function returns a Qobj with superrep 'super', leading to the different representations
    # Furthermore, manually converting sqrtmA back to choi leads to incorrect calculation of the fidelity
    # Likewise with converting B back to super representation 'super'
    # Therefore the only solution is to suppress the warning
    # Maintains correct fidelity calculation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # We don't actually need the whole matrix here, just the trace
        # of its square root, so let's just get its eigenenergies instead.
        # We also truncate negative eigenvalues to avoid nan propagation;
        # even for positive semidefinite matrices, small negative eigenvalues
        # can be reported.
        eig_vals = (sqrtmA * B * sqrtmA).eigenenergies()
    
    return float(np.real(np.sqrt(eig_vals[eig_vals > 0]).sum()))


def _hilbert_space_dims(oper):
    """
    For a quantum channel `oper`, return the dimensions `[dims_out, dims_in]`
    of the output Hilbert space and the input Hilbert space.
    - If oper is a unitary, then `oper.dims == [dims_out, dims_in]`.
    - If oper is a list of Kraus operators, then
     `oper[0].dims == [dims_out, dims_in]`.
    - If oper is a superoperator with `oper.superrep == 'super'`:
     `oper.dims == [[dims_out, dims_out], [dims_in, dims_in]]`
    - If oper is a superoperator with `oper.superrep == 'choi'`:
     `oper.dims == [[dims_in, dims_out], [dims_in, dims_out]]`
    - If oper is a superoperator with `oper.superrep == 'chi', then
      `dims_out == dims_in` and
      `oper.dims == [[dims_out, dims_out], [dims_out, dims_out]]`.
    :param oper: A quantum channel, represented by a unitary, a list of Kraus
    operators, or a superoperator
    :return: `[dims_out, dims_in]`, where `dims_out` and `dims_in` are lists
     of integers
    """
    if isinstance(oper, list):
        return oper[0].dims
    elif oper.type == 'oper':  # interpret as unitary quantum channel
        return oper.dims
    elif oper.type == 'super' and oper.superrep in ['choi', 'chi', 'super']:
        return [oper.dims[0][1], oper.dims[1][0]]
    else:
        raise TypeError('oper is not a valid quantum channel!')


def _process_fidelity_to_id(oper):
    """
    Internal function returning the process fidelity of a quantum channel
    to the identity quantum channel.
    Parameters
    ----------
    oper : :class:`qutip.Qobj`/list
        A unitary operator, or a superoperator in supermatrix, Choi or
        chi-matrix form, or a list of Kraus operators
    Returns
    -------
    fid : float
    """
    dims_out, dims_in = _hilbert_space_dims(oper)
    if dims_out != dims_in:
        raise TypeError('The process fidelity to identity is only defined '
                        'for dimension preserving channels.')
    d = np.prod(dims_in)
    if isinstance(oper, list):  # oper is a list of Kraus operators
        return np.sum([np.abs(k.tr()) ** 2 for k in oper]) / d ** 2
    elif oper.type == 'oper':  # interpret as unitary
        return np.abs(oper.tr()) ** 2 / d ** 2
    elif oper.type == 'super':
        if oper.superrep == 'chi':
            return oper[0, 0].real / d ** 2
        else:  # oper.superrep is either 'super' or 'choi':
            return qt.to_super(oper).tr().real / d ** 2


def _kraus_or_qobj_to_choi(oper):
    if isinstance(oper, list):
        return qt.kraus_to_choi(oper)
    else:
        return qt.to_choi(oper)

def purity_density_matrix(rho : qt.Qobj, d : int):
    """
    Calculate the purity of the density matrix rho.
    The purity of a density matrix is defined as Tr(rho^2).

    Parameters:
    -----------
    rho : qutip.Qobj
        -- the density matrix for which the purity is to be calculated
    d : int
        -- the dimension of the Hilbert space
    
    Returns:
    --------
    purity : float
        -- the purity of the density matrix rho

    Raises:
    -------
    TypeError:
        -- if rho is not a Qobj
    ValueError:
        -- if rho is not a square matrix of dimension d, or
        -- if rho is not unit trace
    
    """
    # Check if rho is a Qobj
    if not isinstance(rho, qt.Qobj):
        raise TypeError("rho must be a Qobj")    
    # Check if rho is a square matrix of dimension d
    if rho.shape != (d, d):
        raise ValueError("rho must be a square matrix of dimension d") 
    # Check if rho is unit trace
    if not np.isclose(rho.tr(), 1.0):
        raise ValueError("rho must be unit trace")
    
    # Initialise return variables
    purity : float = 0.0

    # Calculate the purity of the density matrix rho
    purity = np.real(np.trace(rho * rho))

    return purity

def purity_density_matrix_wrapper(gate_idx, state_idx, gamma_idx, rho, d):

    purity = purity_density_matrix(rho, d)

    return gate_idx, state_idx, gamma_idx, purity


def max_coherences_density_matrix(rho):

    return np.abs(rho[0, -1])

def coherences_density_matrix(rho) -> np.ndarray:

    coherences = np.abs(rho[np.triu_indices(rho.shape[0], k=1)])

    return coherences

def average_coherences_density_matrix(rho):

    coherences = coherences_density_matrix(rho)

    return np.mean(coherences)

def average_coherences_density_matrix_wrapper(gate_idx, state_idx, gamma_idx, rho):

    average_coherences = average_coherences_density_matrix(rho)

    return gate_idx, state_idx, gamma_idx, average_coherences

def evolve_propagator(pulses, collapse_operators, times, options_mesolve, initial_state = None, no_gate = False):    

    dimension = pulses.n_channels + 1

    if initial_state is None:
        # generate random density matrix
        initial_state = qt.rand_dm(dimension)

    if isinstance(times, float):    
        tlist = np.array([times], dtype = np.float64)
    else:
        tlist = times

    U = propagator(tlist, pulses, options_mesolve, c_ops=collapse_operators, no_gate = no_gate)[0]

    final_density_matrix = U(initial_state)

    return final_density_matrix

def evolve_density_matrix(pulses, collapse_operators, times, options_mesolve, initial_state = None, no_gate = False):

    if no_gate:
        pulse_hamiltonian = pulses.gen_0()
    else:
        pulse_hamiltonian = pulses.gen_H()

    dimension = pulses.n_channels + 1    

    if initial_state is None:
        initial_state = qt.rand_ket(dimension)  # Generate a random initial state

    if isinstance(times, float):    
        tlist = np.array([0, times], dtype = np.float64)
    else:
        tlist = times

    # Solve the master equation using mesolve at the specified time
    result = qt.mesolve(pulse_hamiltonian, initial_state, tlist, c_ops = collapse_operators, options = options_mesolve)

    # The result object contains the time-evolved state at the specified time
    final_density_matrix = result.states[-1]

    return final_density_matrix

def chrestenson_unitary(d):
    
    idx = np.arange(d, dtype = np.complex128)
    
    u = idx[:, np.newaxis] * idx[np.newaxis, :]
    
    u *= 2j * np.pi / d
    
    np.exp(u, out = u)
    
    u /= np.sqrt(d)
    
    u = qt.Qobj(u)
    
    return u

# deprecated
def linspace_log(minval, maxval, n):
    
    lin = np.linspace(1.0 * np.log10(minval), 1.0 * np.log10(maxval), n + 1)
    
    lin = lin[1 : ]
    
    lin = 10.0**lin
    
    return lin


def noisevals(cfg, noise_type, noise_threshold: bool = False):

    dim = int(cfg.get_keyval('qudit Settings', 'dimens'))
    
    section = '{} Noise'.format(noise_type)
    
    noise_name = noise_type.replace('ENVT', '').replace('CTRL', '').strip()
    
    flag = eval(cfg.get_keyval(section, '{}_fla'.format(noise_name)))
    
    flag_log = eval(cfg.get_keyval(section, '{}_log'.format(noise_name)))
    
    n = int(cfg.get_keyval(section, '{}_num'.format(noise_name)))
    
    minval = float(cfg.get_keyval(section, '{}_min'.format(noise_name)))
    
    maxval = float(cfg.get_keyval(section, '{}_max'.format(noise_name)))
    
    vals = []
    
    if (flag):
        
        if (flag_log):
        
            vals = np.geomspace(minval, maxval, n).tolist()
            
        else:
            
            vals = np.linspace(minval, maxval, n).tolist()

    gamma_threshold = 0.01

    # if threshold flag is set, filter out noise values above threshold

    if noise_threshold:
    
        noise_vals = list(filter(lambda x: x <= gamma_threshold, vals))

    else:
        
        noise_vals = vals

    # now take every second value from the list noise_vals
    #noise_vals = noise_vals[::2]
        
    return noise_vals

def noise_sections(cfg):

    noise_sections = []

    sections = cfg.get_sections()               

    for section in sections:

        if "Noise" in section:

            noise_sections.append(section.replace(" Noise", ""))

    return noise_sections

def max_gatetime(d: int, safety_factor: float = 1.1) -> int:
    """
    
    Calculate the maximum gate time for a given qudit dimension.
    
    Parameters:
    d (int): qudit dimension
    safety_factor (float): safety factor to multiply by

    Returns:
    max_gatetime (int): maximum gate time for given qudit dimension

    Last edited: 2021-07-28

    """

    # fitted values for max gate time

    x_0 = -3.709970256438777

    x_1 = +3.713116927215376

    x_2 = +0.517738538495380   

    # calculate gate time

    gate_time = x_2 * d**2 + x_1 * d**1 + x_0 * d**0

    # multiply by safety factor

    max_gatetime = gate_time * safety_factor

    # round value to nearest integer

    max_gatetime = int(np.round(max_gatetime))

    return max_gatetime

def GUE_distribution(s):
    return (32/np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

# Parallelized function to compute eigenvalues
def compute_eigenvalues(u) -> np.ndarray:
    return np.linalg.eigvals(u)

def get_eigenvalue_phase_and_spacing_distributions(n_eigenvalues : int = 500000, dim : int = 2, method : str = 'circular') -> tuple[np.ndarray, np.ndarray]:

    print(f"Total number of eigenvalues: {n_eigenvalues}")

# Set alias dimension for the quantum system
    d = dim    

    print(f"Dimension: {d}")

# Number of random quantum gates we want to generate and analyze
    n_gates = int(n_eigenvalues / d)

    print(f"Number of gates: {n_gates}")

# Number of cores to use for parallel processing
    n_jobs = 10  # Use all available cores

    print(f"Number of jobs: {n_jobs}")

    batch_size = int(n_gates / n_jobs)

    print(f"Batch size: {batch_size}")

# START TIME
    start_time = time.time()

    print(f"Start time: {start_time - start_time}")
    print(f"Beginning to generate {n_gates} random {d}x{d} unitary matrices using the {method} method...")

# Generate a list of random unitary matrices
    gates, hamiltonians = generate_gates_and_hamiltonians(n_gates, dim, method = method, n_jobs = n_jobs, batch_size = batch_size, is_super = False) 

# GATE GENERATION TIME
    gate_generation_time = time.time() - start_time

    print(f"Gate generation time: {gate_generation_time}")

# Analyse the eigenvalues of the gates

    print(f"Beginning eigenvalue analysis...")

# Parallel computation of eigenvalues
    eigenvalues = Parallel(n_jobs=n_jobs, batch_size = batch_size, timeout = 10)(delayed(compute_eigenvalues)(u) for u in gates) # type: ignore

# EIGENVALUE ANALYSIS TIME

    eigenvalue_analysis_time = time.time() - gate_generation_time - start_time

    print(f"Eigenvalue analysis time: {eigenvalue_analysis_time}")

# Extract phases
    eigenvalue_phases = np.angle(eigenvalues) # type: ignore

# sort the eigenvalues from smallest to largest
    eigenvalue_phases = np.sort(eigenvalue_phases)

# generate the normalised spacing distribution of adjacent eigenvalues
    spacing_distribution =  np.diff(eigenvalue_phases)

# normalise the spacing distribution
    normalised_spacings = spacing_distribution / np.mean(spacing_distribution, axis = 0)

# concatenate the eigenvalues and normalised spacings
    concatenated_eigenvalues = np.concatenate(eigenvalue_phases)
    concatenated_normalised_spacings = np.concatenate(normalised_spacings)

# EIGENVALUE TIME
    eigenvalue_time = time.time()

    print(f"Total time: {eigenvalue_time - start_time}")

    return concatenated_eigenvalues, concatenated_normalised_spacings

def power_law_model_no_c(x, a,  b, d):
    return a * np.power(x - d, -b)

def exp_decay_model(x, a, b, c, d):
    return d - a * np.exp(-b * (x - c))

def sigmoid_model(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

        