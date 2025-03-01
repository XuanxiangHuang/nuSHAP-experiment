import pandas as pd


if __name__ == '__main__':
    ds_list = ['adult', 'corral', 'iris', 'mux6', 'connect_4', 'spambase', 'spectf', 'clean1', 'coil2000', 'dna', 'mnist']
    runtime_data = {
        'SHAP Runtime': [],
        'nuSHAP Runtime': [],
        'nuSHAP #Samples': []
    }

    for ds in ds_list:
        ds = ds.strip()
        print(f"################## {ds} ##################")

        df_shap_runtime = pd.read_csv(f'runtime_res/shap/{ds}.csv')
        df_nushap_runtime = pd.read_csv(f'runtime_res/nushap/{ds}.csv')

        shap_runtime = df_shap_runtime['runtime'].mean()
        nushap_runtime = df_nushap_runtime['runtime'].mean()
        nushap_samples = df_nushap_runtime['n_samples'].mean()

        runtime_data['SHAP Runtime'].append(shap_runtime)
        runtime_data['nuSHAP Runtime'].append(nushap_runtime)
        runtime_data['nuSHAP #Samples'].append(nushap_samples)

        print(f"SHAP runtime: {df_shap_runtime['runtime'].mean():.1f} seconds")
        print(f"nuSHAP runtime & #samples: {df_nushap_runtime['runtime'].mean():.1f} seconds, {df_nushap_runtime['n_samples'].mean():.1f} samples")

    # Runtime table
    runtime_df = pd.DataFrame(runtime_data, index=ds_list).T
    runtime_df.index = ['SHAP Runtime', 'nuSHAP Runtime', 'nuSHAP #Samples']

    # Convert runtime data to LaTeX table
    latex_runtime_table = runtime_df.to_latex(index=True, escape=False, float_format="%.1f")
    print("\nLaTeX Table for Runtime and #Samples:")
    print(latex_runtime_table)
