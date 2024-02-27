from pathlib import Path
import sys
import logging
import os
import pandas as pd

# Adding path to import cleanSEEG
path = str(Path(Path(__file__).parent.absolute()).parent.parent.absolute())
print(path)
# sys.path.append(path)

# Import cleanSEEG
# from interictal_classifier import utils, model, data_setup

import torch
sys.path.insert(0,'/scratch/mcesped/code/NoiseDetection_iEEG/interictal_classifier/')
import model
import data_setup
import utils
import json
import numpy as np

def classify_matrix(y_pred, thr_1, thr_2, artifact_preference = False):
    if artifact_preference:
        matrix = np.empty(y_pred.shape)
        matrix[:, 0] = (y_pred[:, 0] > thr_1).astype(int)
        matrix[:, 1] = (y_pred[:, 1] > thr_2).astype(int)
        # Check conditions using vectorized operations
        condition_phys = np.logical_and(matrix[:, 0] == 0, matrix[:, 1] == 0)
        condition_art = np.logical_and(matrix[:, 0] == 1, matrix[:, 1] == 0)
        condition_path = np.logical_and(matrix[:, 0] == 0, matrix[:, 1] == 1)
        condition_undefined = np.logical_and(matrix[:, 0] == 1, matrix[:, 1] == 1)
        
            # Initialize an array to store results
        results = np.empty(matrix.shape[0])  # Adjust the dtype according to the length of the longest label
        
        # Assign labels based on conditions
        results[condition_phys] = 2
        results[condition_art] = 0
        results[condition_path] = 1
        results[condition_undefined] = 0
        return results
    # Use argmax
    results = np.argmax(y_pred, axis=1)
    # Update those below thr
    condition_clean = np.logical_and(y_pred[:, 0] <= thr_1, y_pred[:, 1] <= thr_2)
    results[condition_clean] = 2  
    return results

def main():
    df_train_path = snakemake.input.df_train
    df_val_path = snakemake.input.df_val
    model_paths = snakemake.input.model_paths
    results_paths = snakemake.input.results_paths
    processes = int(snakemake.threads)
    out_results = snakemake.output.out_results
    LOG_FILENAME = snakemake.log[0]
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    try:
        # load dataset
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        # Zip files and transforms
        features = snakemake.config['feature']
        srate = snakemake.config['srate']
        categories = snakemake.config['category']
        architecture = snakemake.config['architecture']
        zip_files = [
            f"/home/mcesped/scratch/Datasets/{srate}Hz/Dataset_Fnusa_Combined.zip",
            f"/home/mcesped/scratch/Datasets/{srate}Hz/Dataset_Mayo_Combined.zip",
        ]
        
        concat_y_pred = []
        thresholds = []
        for model_path, category, results_path in zip(model_paths, categories, results_paths):
            (
                train_dataloader,
                val_dataloader,
                _,
                ) = data_setup.create_dataloaders_uncompress(
                zip_files,
                df_train_path,
                df_val_path,
                None,
                None,
                batch_size=32,
                num_workers=processes,
                dataset_class="SpectrogramDir",
                binary=True,
                previosly_uncompressed = False,
                features=features,
                binary_cat=category
                )
            # Create model
            input_size_map = {
                'CWT': 1,
                'Combined': 2,
                'Hilbert': 1
                }
            input_size = input_size_map[features] # Number of images per example
            input_length_map = {
            2048: 6094,
            1024: 3046
            }
            input_length =  input_length_map[srate]
            n_classes = 1
            models = {
            "CNN_Long_Data": model.CNN_Long_Data(n_classes=n_classes, input_size=input_size, input_length=input_length), # 35,516
            "CNN_RNN_Long_Data": model.CNN_RNN_Long_Data(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_Long_Data2": model.CNN_Long_Data2(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_Long_Data3": model.CNN_Long_Data3(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_Long_Data4": model.CNN_Long_Data4(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_Long_Data5": model.CNN_Long_Data5(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_Resnet": model.custom_resnet34(n_classes = n_classes, input_size=input_size),
            "CNN_RNN_Long_Data1": model.CNN_RNN_Long_Data1(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_RNN_Long_Data2": model.CNN_RNN_Long_Data2(n_classes=n_classes, input_size=input_size, input_length=input_length),
            "CNN_RNN_Long_Data3": model.CNN_RNN_Long_Data3(n_classes=n_classes, input_size=input_size, input_length=input_length),
            }
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_0 = models[architecture].to(device)
            
            model_0.load_state_dict(torch.load(model_path))

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Put model in eval mode
            model_0.eval()
            # Initiate values
            y_total, y_pred_total = np.array([]), np.array([])

            # Turn on inference context manager
            with torch.inference_mode():
                # Loop through DataLoader batches
                for batch, (X, y) in enumerate(val_dataloader):
                    # Send data to target device
                    X, y = X.to(device), y.to(device)

                    # 1. Forward pass
                    test_pred_logits = model_0(X).squeeze()
                    
                    # Apply softmax
                    y_pred_prob = torch.sigmoid(test_pred_logits)

                    # Save results
                    #y_total = np.concatenate([y_total, y.detach().numpy()])
                    y_pred_total = np.concatenate(
                        [y_pred_total, y_pred_prob.detach().numpy()]
                    )
            concat_y_pred.append(y_pred_total)
            # Load threshold
            with open(results_path, 'r') as f:
                data = json.load(f)
            thresholds.append(data['threshold'])
        # Merge
        concat_y_pred = np.vstack(concat_y_pred)
        # Get true y
        df_val = pd.read_csv(df_val_path, sep=',')
        y_true = df_val.category_id.values
        
        # Run metrics
        results_val = dict()
        # First giving more weight to artifacts
        merged_y_pred = classify_matrix(concat_y_pred.T, thresholds[0], thresholds[1], artifact_preference=True).astype(int)
        results_val['artifact_pref'] = utils.classication_metrics(y_true, merged_y_pred, test=True, output_dict=True)
        # With argmax
        merged_y_pred = classify_matrix(concat_y_pred.T, thresholds[0], thresholds[1], artifact_preference=False).astype(int)
        results_val['argmax'] = utils.classication_metrics(y_true, merged_y_pred, test=True, output_dict=True)
        print(results_val['artifact_pref'], flush=True)
        print(results_val['argmax'], flush=True)
        # Save them
        with open(out_results, "w") as write_file:
                json.dump(results_val, write_file)
    
    except:
        logging.exception('Got exception on main handler')
        raise

if __name__=="__main__":
    main()