import numpy as np
import pandas as pd

np.random.seed(117)
NUM_ROWS = 50

if __name__ == '__main__':
    name_list = ['adult', 'corral', 'iris', 'mux6', 'connect_4', 'spambase', 'spectf', 'clean1', 'coil2000', 'dna']
    for ds in name_list:
        df = pd.read_csv(f'datasets/{ds}.csv', sep=',')
        target_column = df.columns[-1]  # Get the last column
        num_classes = df[target_column].nunique()  # Count unique values in the last column
        print(f"################## {ds}({df.shape[1] - 1, num_classes}) ##################")
        X = df.iloc[:, :-1].to_numpy()
        total_rows = df.shape[0]
        if ds == 'mnist':
            random_indices = np.random.choice(total_rows, 5, replace=True)
        else:
            random_indices = np.random.choice(total_rows, NUM_ROWS, replace=False)
        output_file = f'exp_idx/{ds}.csv'
        random_indices_df = pd.DataFrame(random_indices, columns=['Random_Index'])
        random_indices_df.to_csv(output_file, index=False)
