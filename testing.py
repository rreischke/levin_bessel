import pylevin as levin
import numpy as np
import time
import matplotlib.pyplot as plt
import hankel
from hankel import HankelTransform     # Import the basic class

k = np.linspace(0.1,110,100) 
N=1
integral_type = 0
N_thread = 8 # Number of threads used for hyperthreading
logx = True # Tells the code to create a logarithmic spline in x for f(x)
logy = True # Tells the code to create a logarithmic spline in y for y = f(x)
lp_single = levin.pylevin(integral_type, k, (1/k)[:,None], logx, logy, N_thread) #Constructor of the class
