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
    zip_files = [
        "/home/mcesped/scratch/Datasets/Dataset_Fnusa_np_DWT.zip",
        "/home/mcesped/scratch/Datasets/Dataset_Mayo_np_DWT.zip",
    ]
    df_path = '/scratch/mcesped/Datasets/segments_mayo_fnusa_curated.csv'

    # Create dataloaders
    train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders_tree(
        zip_files, df_path, batch_size=-1, num_workers=32
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
        n_estimators=6000,  # How many trees to use, the more the better, but decrease learning rate if many used.
        verbosity=1,  # If to show more errors or not.
        objective="multi:softmax",  # Type of target variable.
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
    param_grid = dict({'n_estimators': [6000, 7000],
                    'max_depth': [2, 3, 4],
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

def main(model_name: str, dir_save: str):
    print("start", end="\n", flush=True)
    print("torch", end="\n", flush=True)
    import torch
    from xgboost import XGBClassifier
    import sys
    import model
    import data_setup
    from engine_tree import train
    import utils
    import torch.nn as nn
    import json

    print("finish import", end="\n", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    # stdout_fileno = sys.stdout
    # stdout_fileno.write('test\n')
    print("test", end="\n", flush=True)

    # Zip files
    zip_files = [
        "/home/mcesped/scratch/Datasets/Dataset_Fnusa_np.zip",
        "/home/mcesped/scratch/Datasets/Dataset_Mayo_np.zip",
    ]
    df_path = '/scratch/mcesped/Datasets/segments_mayo_fnusa_curated.csv'

    # Create dataloaders
    train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders_tree(
        zip_files, df_path, batch_size=1024, num_workers=32
    )

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    # Create model
    model_0 = XGBClassifier(
        max_depth=3,  # Depth of each tree
        learning_rate=0.1,            # How much to shrink error in each subsequent training. Trade-off with no. estimators.
        n_estimators=200,  # How many trees to use, the more the better, but decrease learning rate if many used.
        verbosity=1,  # If to show more errors or not.
        objective="multi:softmax",  # Type of target variable.
        booster="gbtree",  # What to boost. Trees in this case.
        n_jobs=32,  # Parallel jobs to run. Set your processor number.
        random_state=20201107,  # Seed
        gamma=0.001,                  # Minimum loss reduction required to make a further partition on a leaf node of the tree. (Controls growth!)
        subsample=0.632,              # Subsample ratio. Can set lower
        colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
        colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
        colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
    )
    print(f"Model: {model_0}")
    # Load model if it was trained before
    # if os.path.exists(os.path.join(dir_save, "checkpoint.model")):
    #     # load the last checkpoint with the best model
    #     print("Previous trained model found. Loading... \n", flush=True)
    #     model_0.load_model(os.path.join(dir_save, "checkpoint.model"))

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model_0
    model_0_results = train(
        model=model_0,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        epochs=NUM_EPOCHS,
        dir_save=dir_save,
    )

    # Save results
    path_results = os.path.join(dir_save, "results.txt")
    with open(path_results, "w") as write_file:
        json.dump(model_0_results, write_file)

    # End the timer and stdout_fileno.write out how long it took
    end_time = timer()
    print(
        f"[INFO] Total training time: {end_time-start_time:.3f} seconds",
        end="\n",
        flush=True,
    )

    # Save the model
    # save_model(model=model_0,
    #            target_dir="models",
    #            model_name="05_going_modular_cell_mode_tinyvgg_model.pth")


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