import pandas as pd
import numpy as np
import timeit
from kernel_fca_oo import LexiSystem, LexiTSystem

c_i = range(0, 12)
KATALOOG = "..\\"

fns = ["cmp-bin-zoo.csv",
       "cmp-bin-house-votes-84.csv",
       "cmp-bin-student-gradings.csv"]

for fn in fns:
    print("\nFile:", fn)
    data = pd.read_csv(KATALOOG+fn, sep=',', index_col=0, encoding='latin1')
    result_df = pd.DataFrame(columns = ["time"] + ["c"+str(i) for i in c_i])
    systems = {"CLFT": LexiTSystem(data, transform="CL", full_lexi=True),
               "FLFT": LexiTSystem(data, transform="FL", full_lexi=True),
               "CLF": LexiSystem(data, transform="CL", full_lexi=True),
               "FLF": LexiSystem(data, transform="FL", full_lexi=True),
               "CL": LexiSystem(data, transform="CL", full_lexi=False),  # Probably simplest and best version
               "FL": LexiSystem(data, transform="FL", full_lexi=False),
               "CLT": LexiTSystem(data, transform="CL", full_lexi=False),
               "FLT": LexiTSystem(data, transform="FL", full_lexi=False)}
    ones = data.sum().sum()
    for s_name, system in systems.items():
        start = timeit.default_timer()
        _, uc =  system.conceptchaincover(uncovered=0.1, max_cc=20)
        time = timeit.default_timer() - start
        if len(uc) < len(c_i):
            uc += [np.nan] * (len(c_i) - len(uc))  
        result_df.loc[s_name] = [time] + uc[:len(c_i)]
    print(result_df)
    result_df.to_csv(path_or_buf=KATALOOG+"res-"+fn)
