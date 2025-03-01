import pandas as pd
import numpy as np

if __name__ == '__main__':
    ds_list = ['adult', 'corral', 'iris', 'mux6', 'connect_4', 'spambase', 'spectf', 'clean1', 'coil2000', 'dna', 'mnist']
    summary_table = []

    for ds in ds_list:
        ds = ds.strip()
        print(f"################## {ds} ##################")
        # Load the RBO scores CSV
        df_rbo = pd.read_csv(f'rbo_sc/{ds}.csv')

        # Collect statistics for nuSHAP-SHAP
        min_val_0 = df_rbo['nuSHAP-SHAP'].min()
        max_val_0 = df_rbo['nuSHAP-SHAP'].max()
        mean_val_0 = df_rbo['nuSHAP-SHAP'].mean().round(2)

        # Collect statistics for nuSHAP-abs(SHAP)
        min_val_1 = df_rbo['nuSHAP-abs(SHAP)'].min()
        max_val_1 = df_rbo['nuSHAP-abs(SHAP)'].max()
        mean_val_1 = df_rbo['nuSHAP-abs(SHAP)'].mean().round(2)

        # Append the statistics to the summary table
        summary_table.append({
            "Dataset": ds,
            "Metric": "nuSHAP-SHAP",
            "Min": min_val_0,
            "Max": max_val_0,
            "Mean": mean_val_0
        })
        summary_table.append({
            "Dataset": ds,
            "Metric": "nuSHAP-abs(SHAP)",
            "Min": min_val_1,
            "Max": max_val_1,
            "Mean": mean_val_1
        })

    # Convert summary statistics to a DataFrame for display
    summary_df = pd.DataFrame(summary_table)

    # Ensure the 'Dataset' column retains its order
    summary_df['Dataset'] = pd.Categorical(summary_df['Dataset'], categories=ds_list, ordered=True)

    # Pivot the table to get a proper format for transposition
    pivot_df = summary_df.pivot(index="Dataset", columns="Metric", values=["Min", "Max", "Mean"])

    # Transpose the DataFrame (flip rows and columns)
    transposed_df = pivot_df.T

    # Save the transposed DataFrame to CSV
    transposed_df.to_csv('rbo_sc/rbo_summary.csv')

    # Print the summary table
    print("\nSummary Table (Transposed):")
    print(transposed_df)
