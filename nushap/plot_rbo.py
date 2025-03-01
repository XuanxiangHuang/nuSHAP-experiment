import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    ds_list = ['adult', 'corral', 'iris', 'mux6', 'connect_4', 'spambase', 'spectf', 'clean1', 'coil2000', 'dna', 'mnist']

    for ds in ds_list:
        ds = ds.strip()
        print(f"################## {ds} ##################")
        # Load the RBO scores CSV
        df_rbo = pd.read_csv(f'rbo_sc/{ds}.csv')

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Plot nuSHAP-SHAP histogram on the first subplot
        sns.histplot(df_rbo['nuSHAP-SHAP'], bins=20, kde=True, ax=axes[0], color='blue')
        axes[0].set_xlabel('')  # Remove x-axis label
        axes[0].set_ylabel('')  # Remove y-axis label

        # Plot nuSHAP-abs(SHAP) histogram on the second subplot
        sns.histplot(df_rbo['nuSHAP-abs(SHAP)'], bins=20, kde=True, ax=axes[1], color='green')
        axes[1].set_xlabel('')  # Remove x-axis label
        axes[1].set_ylabel('')  # Remove y-axis label

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the plot as an image file (e.g., PNG, JPEG, etc.)
        plt.savefig(f'rbo_figs/{ds}.png', dpi=100)

        # Close the plot to free up memory (if running in a loop or larger dataset)
        plt.close()
