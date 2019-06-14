try:
    from code.schultercore4 import *
except ImportError:
    from schultercore4 import *   # when running from terminal, the directory may not be identified as a package
import os
import matplotlib.pyplot as plt
import sys
import time

params=load_parameters()
print(params)