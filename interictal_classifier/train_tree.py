import os
import argparse

def run_CV():
    import zipfile
    import scipy.io
    import pandas as pd
    import numpy as np
    import os
    import re
    import matplotlib.pyplot as plt
    import mne
    import torch
    from torchvision import datasets, transforms
    import torch
    from xgboost import XGBClassifier
    import sys
    import sys
    import data_setup
    import model
    from engine_tree import train
    import utils
    import torch.nn as nn
    from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
    import json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Zip files
    srate=1024
    zip_files = [
        f"/home/mcesped/scratch/Datasets/Dataset_Fnusa_{srate}.zip",
        f"/home/mcesped/scratch/Datasets/Dataset_Mayo_{srate}.zip",
    ]
    df_train_path = '/scratch/mcesped/Datasets/Noise_detection/df_train_curated.csv'
    df_val_path = '/scratch/mcesped/Datasets/Noise_detection/df_val_curated.csv'

    (
        train_dataloader,
        val_dataloader,
        _,
    ) = data_setup.create_dataloaders_tree(
        zip_files,
        df_train_path,
        df_val_path,
        batch_size=-1,
        num_workers=32,
        binary=True,
        previosly_uncompressed = False,
    )

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    X_train, y_train = next(iter(train_dataloader))
    X_test, y_test = next(iter(val_dataloader))

    X_train, y_train = X_train.detach().numpy(), y_train.detach().numpy()
    X_test, y_test = X_test.detach().numpy(), y_test.detach().numpy()

    # Create model
    model = XGBClassifier(
        max_depth=2,  # Depth of each tree
        learning_rate=0.001,            # How much to shrink error in each subsequent training. Trade-off with no. estimators.
        n_estimators=2500,  # How many trees to use, the more the better, but decrease learning rate if many used.
        verbosity=1,  # If to show more errors or not.
        objective="binary:logistic",   # Type of target variable.
        booster="gbtree",  # What to boost. Trees in this case.
        n_jobs=32,  # Parallel jobs to run. Set your processor number.
        random_state=20201107,  # Seed
        gamma=0.001,                  # Minimum loss reduction required to make a further partition on a leaf node of the tree. (Controls growth!)
        colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
        colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
        colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
    )

    # Crossvalidation object
    cv_object = StratifiedKFold(n_splits=3)
    # Define the parameters. Play with this grid!
    param_grid = dict({'max_depth': [5, 6, 7, 8],
                    'learning_rate' : [0.001, 0.01]
                    })

    # Define grid search object.
    GridXGB = GridSearchCV(model,        # Original XGB. 
                        param_grid,          # Parameter grid
                        cv = cv_object,      # Cross-validation object.  
                        scoring = 'f1_weighted', # How to rank outputs.
                        n_jobs = 32,          # Parallel jobs. -1 is "all you have"
                        refit = False,       # If refit at the end with the best. We'll do it manually.
                        verbose = 2          # If to show what it is doing.
                        )

    GridXGB.fit(X_train, y_train)

    print(GridXGB.best_params_, end='\n', flush=True)

if __name__ == "__main__":
    # Model to use
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", type=str, required=True)
    # parser.add_argument(
    #     "-p", "--path", type=str, required=True
    # )  # Path to save model and results
    # args = parser.parse_args()
    # print("test", end="\n", flush=True)
    # main(args.model, args.path)
    run_CV()