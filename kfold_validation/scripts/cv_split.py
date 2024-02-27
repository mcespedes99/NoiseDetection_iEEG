import numpy as np
# from sklearn.model_selection import GroupKFold
import pandas as pd
from sklearn_repeated_group_k_fold import RepeatedGroupKFold

def main(df, n_splits, n_repeats, dfs_train, dfs_val):
    df_total_cv_curated = pd.read_csv(df, sep=",", index_col="index")
    # Sample data
    X = df_total_cv_curated.segment_id.values  # Your data
    y = df_total_cv_curated.category_id.values  # Your labels
    # Define groups where each group represents a different subject
    groups = df_total_cv_curated.patient_id.values  # Random grouping for illustration

    # Create GroupKFold with a custom generator to ensure groups stay together
    gkf = RepeatedGroupKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=46) # 46 guarantees every class is present in val. Check Datasets.ipynb
    #GroupKFold(n_splits=n_splits)

    # Initialize lists to store train and validation indices
    train_indices_list = []
    val_indices_list = []

    # Generate train and validation indices for each split
    for train_indices, val_indices in gkf.split(X, y, groups=groups):
        train_indices_list.append(train_indices)
        val_indices_list.append(val_indices)

    # Generate dataframes train and val
    for split_idx, (train_indices, val_indices, df_train_path, df_val_path) in enumerate(zip(train_indices_list, val_indices_list, dfs_train, dfs_val)):
        print(f"Split {split_idx+1}:")
        df_train = df_total_cv_curated.iloc[train_indices,:].reset_index(drop=True)
        # df_train = unsample_pathology(0.15, df_train)
        # print("Train indices:", train_indices)
        print("Subjs train:", np.unique(df_train.patient_id.values))
        df_train.to_csv(df_train_path, sep=",", index_label='index')
        # print("Validation indices:", val_indices)
        df_val = df_total_cv_curated.iloc[val_indices,:].reset_index(drop=True)
        print("Subjs val:", np.unique(df_val.patient_id.values))
        df_val.to_csv(df_val_path, sep=",", index_label='index')

def unsample_pathology(target_percentage: float, dataframe):
    # Get percentage to extract based on the target one
    _, (len_noise,len_path,len_phys) = np.unique(dataframe.category_id, return_counts=True)              
    percent_extract = target_percentage*(len_noise + len_phys)/(len_path*(1-target_percentage))

    # Get new dataframe
    df = dataframe.loc[(dataframe.category_id != 1)].reset_index(drop=True)
    # Get path segments without changing the dist
    df_path = dataframe.loc[dataframe.category_id == 1, :].reset_index(drop=True)
    df_path_new = pd.DataFrame(columns= df_path.columns)
    total_new_samples = percent_extract*len(df_path)
    for subj in np.unique(df_path.patient_id):
        df_subj = df_path.loc[df_path.patient_id==subj]
        n_subj = int(total_new_samples*len(df_subj)/len(df_path))
        df_path_new = pd.concat([df_path_new, df_subj.sample(n=n_subj)])
    df = pd.concat([df, df_path_new]).sample(frac=1).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = snakemake.input.df_cv
    folds = int(snakemake.config['cv_fold'])
    repeats = int(snakemake.config['repeats'])
    dfs_train = snakemake.output.dfs_train
    dfs_val = snakemake.output.dfs_val
    main(df, folds, repeats, dfs_train, dfs_val)