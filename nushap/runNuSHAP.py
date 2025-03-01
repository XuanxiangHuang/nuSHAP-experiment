#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Running NuSHAP tool
#
#
################################################################################
import pickle
import sys, time
import subprocess
import pandas as pd
import numpy as np
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
import warnings
# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
################################################################################

np.random.seed(188)


if __name__ == '__main__':
    ds_md = [('adult', 'lr'), ('corral', 'lr'), ('iris', 'dt'), ('mux6', 'dt'),
             ('connect_4', 'knn'), ('spambase', 'knn'), ('spectf', 'knn'),
             ('clean1', 'xgboost'), ('coil2000', 'xgboost'), ('dna', 'xgboost')]
    for (ds, md) in ds_md:
        print(f"################## {ds}, {md} ##################")
        df = pd.read_csv(f'datasets/{ds}.csv', sep=',')
        runtime_df_nushap = pd.read_csv(f"runtime_res/nushap/{ds}.csv")
        # default parameters for nuSHAP
        error = 0.0015
        alpha = 0.015

        if ds in ['clean1','coil2000','dna']:
            error = 0.0015
            alpha = 0.015

        random_indices_df = pd.read_csv(f'exp_idx/{ds}.csv')
        random_indices = random_indices_df['Random_Index'].to_numpy()

        all_nushap_scs = []
        all_nushap_time = []

        for i in random_indices:
            print(f"computing NuSHAP for {i}-th data point")
            time_nushap_start = time.perf_counter()

            command = [
                "python3", "tools/nushap/nushap.py",
                "--dataset", f"nushap_samples/{ds}/shap-sample-{i}.csv",
                "--instfile", f"nushap_samples/{ds}/shap-inst-{i}.csv",
                "--error", f"{error}",
                "--alpha", f"{alpha}"
            ]

            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            for line in result.stdout.splitlines():
                if line.startswith("### Svs:"):
                    nushap_sc_for_pred = line
                    break

            all_nushap_scs.append(np.array(list(map(float, nushap_sc_for_pred.replace("### Svs: ", "").split()))))
            time_nushap_end = time.perf_counter()
            all_nushap_time.append(np.round(time_nushap_end - time_nushap_start, 4))

        header_line = ",".join(list(df.iloc[:, :-1].columns))
        header_line = header_line.lstrip("#")
        np.savetxt(f"results/nushap/{ds}.csv", all_nushap_scs,
                   delimiter=",", header=header_line, comments="", fmt='%.4f')

        runtime_df_nushap["runtime"] = all_nushap_time
        runtime_df_nushap.to_csv(f"runtime_res/nushap/{ds}.csv", index=False)
