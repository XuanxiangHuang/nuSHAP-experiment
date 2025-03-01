import pandas as pd
import numpy as np
import rbo


if __name__ == '__main__':
    ds_list = ['adult', 'corral', 'iris', 'mux6', 'connect_4', 'spambase', 'spectf', 'clean1', 'coil2000', 'dna', 'mnist']

    for ds in ds_list:
        ds = ds.strip()
        print(f"################## {ds} ##################")

        df_shap = pd.read_csv(f'results/shap/{ds}.csv').round(4)
        df_nushap = pd.read_csv(f'results/nushap/{ds}.csv').round(4)
        abs_df_shap = df_shap.abs().round(4)

        # Apply row-wise sorting
        df_nushap_ranked = df_nushap.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
        df_shap_ranked = df_shap.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
        abs_df_shap_ranked = abs_df_shap.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)

        # Compute RBO row-wise
        rbo_shap = []
        rbo_abs_shap = []

        for i in range(len(df_nushap_ranked)):
            nushap_rank = df_nushap_ranked.iloc[i]
            shap_rank = df_shap_ranked.iloc[i]
            abs_shap_rank = abs_df_shap_ranked.iloc[i]

            # Compute RBO for nuSHAP vs. SHAP
            similarity_shap = rbo.RankingSimilarity(nushap_rank, shap_rank)
            rbo_shap.append(similarity_shap.rbo(p=0.5, k=5))

            # Compute RBO for nuSHAP vs. abs(SHAP)
            similarity_abs_shap = rbo.RankingSimilarity(nushap_rank, abs_shap_rank)
            rbo_abs_shap.append(similarity_abs_shap.rbo(p=0.5, k=5))

        # Create a DataFrame for the current dataset's RBO results
        rbo_results = {
            'nuSHAP-SHAP': rbo_shap,
            'nuSHAP-abs(SHAP)': rbo_abs_shap
        }
        rbo_df = pd.DataFrame(rbo_results)

        # Save the RBO results to a CSV file named after the dataset
        rbo_df.to_csv(f'rbo_sc/{ds}.csv', index=False, float_format="%.2f")
        print(f"RBO results for {ds}")
