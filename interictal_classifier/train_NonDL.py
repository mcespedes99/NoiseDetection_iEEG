import os
import pandas as pd
import numpy as np
import argparse
import torch
from xgboost import XGBClassifier
import data_setup
from sklearn_repeated_group_k_fold import RepeatedGroupKFold
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def evaluate_LR(X, y, groups):
    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    # Create model
    model_0 = LogisticRegression(max_iter=10000)
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", model_0)])

    # Crossvalidation object
    cv_object = RepeatedGroupKFold(n_splits=10, n_repeats=1, random_state=46) 
    # Define the parameters. Play with this grid! #'ovr', 
    # https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
    param_grid = {'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  "logistic__penalty":["l2"], # l1 and elastic net don't work on the solve
                  'logistic__solver': ['saga', 'newton-cg'],
                  'logistic__multi_class': ['multinomial']}

    # Define grid search object.
    GridLR = GridSearchCV(pipe,        # Original Model. 
                        param_grid,          # Parameter grid
                        cv = cv_object,      # Cross-validation object.  
                        scoring = 'f1_macro', # How to rank outputs.
                        n_jobs = 32,          # Parallel jobs. -1 is "all you have"
                        refit = False,       # If refit at the end with the best. We'll do it manually.
                        verbose = 2          # If to show what it is doing.
                        )

    GridLR.fit(X, y, groups=groups)

    print(GridLR.best_params_, end='\n', flush=True)

    # Manually refit the model with the best parameters
    best_model = pipe.set_params(**GridLR.best_params_)
    return best_model, GridLR.best_params_

def evaluate_XGBoost(X, y, groups):
    # Create model
    model_0 = XGBClassifier(
        max_depth=2,  # Depth of each tree
        learning_rate=0.001,            # How much to shrink error in each subsequent training. Trade-off with no. estimators.
        n_estimators=2500,  # How many trees to use, the more the better, but decrease learning rate if many used.
        verbosity=1,  # If to show more errors or not.
        objective="multi:softmax",   # Type of target variable.
        booster="gbtree",  # What to boost. Trees in this case.
        n_jobs=32,  # Parallel jobs to run. Set your processor number.
        random_state=20201107,  # Seed
        gamma=0.001,                  # Minimum loss reduction required to make a further partition on a leaf node of the tree. (Controls growth!)
        colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
        colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
        colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
    )

    # Crossvalidation object
    cv_object = RepeatedGroupKFold(n_splits=10, n_repeats=1, random_state=46) 
    # Define the parameters. Play with this grid!
    # https://towardsdatascience.com/mastering-random-forests-a-comprehensive-guide-51307c129cb1
    param_grid = dict({'n_estimators': [500, 750, 1000],
                    'max_depth': [2, 3, 4],
                    'learning_rate' : [0.001, 0.01]
                    })

    # Define grid search object.
    GridXGB = GridSearchCV(model_0,        # Original XGB. 
                        param_grid,          # Parameter grid
                        cv = cv_object,      # Cross-validation object.  
                        scoring = 'f1_macro', # How to rank outputs.
                        n_jobs = 32,          # Parallel jobs. -1 is "all you have"
                        refit = False,       # If refit at the end with the best. We'll do it manually.
                        verbose = 2          # If to show what it is doing.
                        )

    GridXGB.fit(X, y, groups=groups)

    print(GridXGB.best_params_, end='\n', flush=True)

    # Now fit the best model to get CV results
    model_0 = XGBClassifier(
        max_depth=GridXGB.best_params_['max_depth'],  # Depth of each tree
        learning_rate=GridXGB.best_params_['learning_rate'],            # How much to shrink error in each subsequent training. Trade-off with no. estimators.
        n_estimators=GridXGB.best_params_['n_estimators'],  # How many trees to use, the more the better, but decrease learning rate if many used.
        verbosity=1,  # If to show more errors or not.
        objective="multi:softmax",   # Type of target variable.
        booster="gbtree",  # What to boost. Trees in this case.
        n_jobs=32,  # Parallel jobs to run. Set your processor number.
        random_state=20201107,  # Seed
        gamma=0.001,                  # Minimum loss reduction required to make a further partition on a leaf node of the tree. (Controls growth!)
        colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
        colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
        colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
    )
    return model_0, GridXGB.best_params_

def evaluate_RF(X, y, groups):
    # Create model
    model_0 = RandomForestClassifier(n_estimators=1000, # Number of trees to train
                       criterion='entropy', # How to train the trees. Also supports entropy.
                       class_weight='balanced',
                        max_depth=14,
                        n_jobs=32)

    # Crossvalidation object
    cv_object = RepeatedGroupKFold(n_splits=10, n_repeats=1, random_state=46) 
    # Define the parameters. Play with this grid!
    param_grid = dict({'n_estimators': [250, 500, 750, 1000],
                       'max_depth': [3, 5, 7],
                       'max_features' : [0.5, 0.75],
                       'min_samples_leaf': [0.05],
                       'criterion': ['gini', 'entropy']
                       })

    # Define grid search object.
    GridRF = GridSearchCV(model_0,        # Original XGB. 
                        param_grid,          # Parameter grid
                        cv = cv_object,      # Cross-validation object.  
                        scoring = 'f1_macro', # How to rank outputs.
                        n_jobs = 32,          # Parallel jobs. -1 is "all you have"
                        refit = False,       # If refit at the end with the best. We'll do it manually.
                        verbose = 2          # If to show what it is doing.
                        )

    GridRF.fit(X, y, groups=groups)

    print(GridRF.best_params_, end='\n', flush=True)

    # Now fit the best model to get CV results
    model_0 = RandomForestClassifier(n_estimators=GridRF.best_params_['n_estimators'], # Number of trees to train
                       criterion=GridRF.best_params_['criterion'], # How to train the trees. Also supports entropy.
                       max_depth=GridRF.best_params_['max_depth'],
                       max_features=GridRF.best_params_['max_features'],
                       min_samples_leaf=GridRF.best_params_['min_samples_leaf'],
                       class_weight='balanced',
                        n_jobs=32)
    return model_0, GridRF.best_params_

def run_CV(algorithm: str, dir_save: str, srate: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Zip files
    zip_files = [
        f"/home/mcesped/scratch/Datasets/{srate}Hz/Dataset_Fnusa_DWT.zip",
        f"/home/mcesped/scratch/Datasets/{srate}Hz/Dataset_Mayo_DWT.zip",
    ]
    df_train_path = '/scratch/mcesped/Datasets/df_total_cv_curated_new.csv'
    df_val_path = None

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
        previosly_uncompressed = False,
    )

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    X_train, y_train = next(iter(train_dataloader))

    X_train, y_train = X_train.detach().numpy(), y_train.detach().numpy()

    df_train = train_dataloader.dataset.df
    groups = df_train.patient_id.values

    # Get best model
    map_to_func = {
        'XGB': evaluate_XGBoost,
        'RF': evaluate_RF,
        'LR': evaluate_LR
    }
    model_0, best_params = map_to_func[algorithm](X_train, y_train, groups)

    # Save params
    path_results = os.path.join(dir_save, "best_params.txt")
    with open(path_results, "w") as write_file:
        json.dump(best_params, write_file)    
    
    # Run CV
    cv_object = RepeatedGroupKFold(n_splits=10, n_repeats=5, random_state=46) 
    cv_results = cross_validate(model_0, X_train, y_train, groups=groups, cv=cv_object, 
                                scoring=['accuracy', 'balanced_accuracy', 'f1_macro',
                                         'precision_macro', 'recall_macro'])
    print('\n Best Results:\n', cv_results)

    # Convert to lists
    new_results = dict()
    for key in cv_results.keys():
        new_results[key] = cv_results[key].tolist()

    # Save results
    path_results = os.path.join(dir_save, "results_val.txt")
    with open(path_results, "w") as write_file:
        json.dump(new_results, write_file)

if __name__ == "__main__":
    # Model to use
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", type=str, required=True)
    parser.add_argument(
        "-p", "--path", type=str, required=True
    )  # Path to save model and results
    parser.add_argument(
        "-s", "--srate", type=int, required=True
    )
    args = parser.parse_args()
    # print("test", end="\n", flush=True)
    # main(args.model, args.path)
    run_CV(args.algorithm, args.path, args.srate)