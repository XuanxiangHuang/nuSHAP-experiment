#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Using NuSHAP tool
#
#
################################################################################
import pickle
import sys, time
import pandas as pd
import numpy as np
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
import warnings
from nuSHAP_wrapper import NuSHAP_tabular
# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
################################################################################

np.random.seed(188)


def handle_shap_scores(predictions, vals):
    shap_values_for_preds = []
    for i, pred in enumerate(predictions):
        if shap_sc.values.ndim == 3:
            shap_values_for_preds.append(vals[i, :, pred])
        else:
            shap_values_for_preds.append(vals[i, :])
    return np.array(shap_values_for_preds)


if __name__ == '__main__':
    ds_md = [('adult', 'lr'), ('corral', 'lr'), ('iris', 'dt'), ('mux6', 'dt')]
    for (ds, md) in ds_md:
        print(f"################## {ds}, {md} ##################")
        df = pd.read_csv(f'datasets/{ds}.csv', sep=',')
        train_df, test_df = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=41)
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        # default parameters for nuSHAP
        error = 0.0015
        alpha = 0.015

        if md == 'lr':
            model = sklearn.linear_model.LogisticRegression(max_iter=5000)
            model.fit(X_train, y_train)
        elif md == 'dt':
            model = sklearn.tree.DecisionTreeClassifier(random_state=0)
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Invalid model: {md}")

        my_nushap = NuSHAP_tabular(model.predict_proba, X_train, feature_names=list(train_df.iloc[:, :-1].columns),
                                   keep_index=True)
        explainer = shap.ExactExplainer(model=my_nushap.predict_and_collect_samples, masker=X_train,
                                        feature_names=my_nushap.data_feature_names)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        random_indices_df = pd.read_csv(f'exp_idx/{ds}.csv')
        random_indices = random_indices_df['Random_Index'].to_numpy()

        X = df.iloc[:, :-1].to_numpy()

        all_shap_scs = []
        all_nushap_scs = []
        all_shap_time = []
        all_nushap_time = []
        all_nushap_samples = []

        for i in random_indices:
            x = X[i]
            inst_x = x.reshape(1, -1)
            pred = model.predict(inst_x)
            print(f"{i}-th data point, prediction: {pred}")

            time_shap_start = time.perf_counter()
            shap_sc = explainer(inst_x)
            shap_scs_for_pred = handle_shap_scores(pred, shap_sc.values)
            print("SHAP scores: ", shap_scs_for_pred)
            all_shap_scs.append(shap_scs_for_pred.flatten())
            time_shap_end = time.perf_counter()
            all_shap_time.append(np.round(time_shap_end - time_shap_start, 4))

            time_nushap_start = time.perf_counter()
            n_samples, nushap_sc_for_pred = my_nushap.comp_nushap(inst_x, pred,
                                                                  f"nushap_samples/{ds}/shap-sample-{i}.csv",
                                                                  f"nushap_samples/{ds}/shap-inst-{i}.csv", error, alpha)
            all_nushap_scs.append(np.array(list(map(float, nushap_sc_for_pred.replace("### Svs: ", "").split()))))
            time_nushap_end = time.perf_counter()
            all_nushap_time.append(np.round(time_nushap_end - time_nushap_start, 4))
            all_nushap_samples.append(n_samples)

        header_line = ",".join(my_nushap.data_feature_names)
        header_line = header_line.lstrip("#")
        np.savetxt(f"results/shap/{ds}.csv", all_shap_scs,
                   delimiter=",", header=header_line, comments="", fmt='%.4f')
        np.savetxt(f"results/nushap/{ds}.csv", all_nushap_scs,
                   delimiter=",", header=header_line, comments="", fmt='%.4f')

        runtime_df_shap = pd.DataFrame(all_shap_time, columns=["runtime"])
        runtime_df_shap.to_csv(f"runtime_res/shap/{ds}.csv", index=False)
        runtime_df_nushap = pd.DataFrame({
            "runtime": all_nushap_time,
            "n_samples": all_nushap_samples
        })
        runtime_df_nushap.to_csv(f"runtime_res/nushap/{ds}.csv", index=False)
