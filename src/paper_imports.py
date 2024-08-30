# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import copy
import datetime
import fnmatch
import importlib
import os
import gc
import re
import sys
import pickle
import time
import functools
import warnings

from itertools import combinations
from math import factorial
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import tikzplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statistics as st
import qutip as qt
import qutip.operators as op
import qutip.control.pulseoptim as cpo
import qutip.logging_utils as logging
import statsmodels.api as sm

from matplotlib.legend_handler import HandlerTuple
from sklearn.linear_model import LinearRegression, RANSACRegressor
from statsmodels.stats.diagnostic import het_breuschpagan
from Cython.Build import cythonize
from cirq.testing.lin_alg_utils import random_unitary, random_special_unitary
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from qutip import Qobj, to_super
from scipy.linalg import fractional_matrix_power
from scipy.optimize import root_scalar, RootResults, curve_fit, minimize_scalar, fsolve, minimize
from scipy.interpolate import CubicSpline
from scipy.stats import linregress, weibull_min, entropy
from setuptools import setup
# ----------------------------------------------------------------------------------------------------------------------------------
# Custom libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import bristol
import qudit

from bristol.ensembles import Circular
from qudit import Qudit
# ----------------------------------------------------------------------------------------------------------------------------------
# importlib.reload calls
# ----------------------------------------------------------------------------------------------------------------------------------
importlib.reload(qudit)
# ----------------------------------------------------------------------------------------------------------------------------------
import qudit
from qudit import *
from qudit import Qudit
# ----------------------------------------------------------------------------------------------------------------------------------
