#FUNCTION TO GENERATE A .CSV FILE FROM .M
#Author: Anton Cobian Iregui

import sys
import os
import pandas as pd
import scipy.io
import argparse

def mat2csv(mat_in, csv_out=None):
    name=None
    if not os.path.isfile(mat_in): #Not found
        print(f"ERROR: No file named {mat_in} found\n")
        return
    if mat_in.lower().endswith(".mat"): #Not a .mat file
        print(f"ERROR: File is not .mat")
        return
    if not csv_out.lower().endswith(".csv"): #Name is invalid
        print(f"ERROR: Ouput name does not include .csv extension")
        return
    if csv_out==None: #Default, keep name
        name=mat_in.remove(".mat")

    try:
        mat=scipy.io.loadmat(mat_in)
    except:
        pass
if __name__=="__main__":
    pass

    

