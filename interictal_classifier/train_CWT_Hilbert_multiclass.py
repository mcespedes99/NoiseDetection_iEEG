import os
import argparse


def main(model_name: str, dir_save: str, processes: int, features: str, srate: int, df_val_path:str, df_train_path:str):
    print("start", end="\n", flush=True)
    print("torch", end="\n", flush=True)
    import torch
    from torchvision import transforms
    import sys
    import model
    import data_setup
    from engine_multiclass import train
    import torch.optim.lr_scheduler as lr_scheduler
    import utils
    import torch.nn as nn
    import json

    print("finish import", end="\n", flush=True)
    assert features in ['CWT','Hilbert', 'Combined'] 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    # stdout_fileno = sys.stdout
    # stdout_fileno.write('test\n')
    print("test", end="\n", flush=True)

    ## PARAMETERS
    input_size_map = {
        'CWT': 1,
        'Combined': 2,
        'Hilbert': 1
    }
    input_size = input_size_map[features] # Number of images per example
    validation_freq = 100 # as a percentage of the train size
    input_length_map = {
        2048: 6094,
        1024: 3046
    }
    # Set number of epochs
    NUM_EPOCHS = 100
    max_iter = None
    input_length =  input_length_map[srate]
    # Zip files and transforms
    zip_files = [
        f"/home/mcesped/scratch/Datasets/{srate}Hz/Dataset_Fnusa_Combined.zip",
        f"/home/mcesped/scratch/Datasets/{srate}Hz/Dataset_Mayo_Combined.zip",
    ]
    # df_path = '/scratch/mcesped/Datasets/segments_mayo_fnusa_curated_big_version.csv'
    # df_train_path = '/scratch/mcesped/Datasets/Multiclass/df_train_curated.csv'
    # df_val_path = '/scratch/mcesped/Datasets/Multiclass/df_val_curated.csv'
    # df_train_path = '/scratch/mcesped/Datasets/Multiclass/df_train_full.csv'
    # df_val_path = '/scratch/mcesped/Datasets/Multiclass/df_val_full.csv'

    train_transform = transforms.Compose(
        [
            #transforms.Resize((30, 100), antialias=True),
            transforms.RandomHorizontalFlip(p=0.3),
            utils.RandomMask(p=0.3, f_max_size= 1, t_max_size = int(0.2*input_length), n_masks_f=1, n_masks_t=2),
        ]
    )
    val_transform = (
        None  # transforms.Compose([transforms.Resize((30, 100), antialias=True)])
    )
    ## FINISH PARAMETERS

    print(f'Using dataframe: {df_train_path} \n', end='\n', flush=True)
    print(f'Using zipfiles: {zip_files} \n', end='\n', flush=True)
    print(f'Using feature: {features} \n', end='\n', flush=True)
    # Create dataloaders
    (
        train_dataloader,
        val_dataloader,
        _,
    ) = data_setup.create_dataloaders_uncompress(
        zip_files,
        df_train_path,
        df_val_path,
        train_transform,
        val_transform,
        batch_size=32,
        features=features,
        num_workers=processes,
        dataset_class="SpectrogramDir",
        binary=False,
        previosly_uncompressed = False
    )
    # data_setup.create_dataloaders_manual(
    #     zip_files,
    #     df_path,
    #     train_transform,
    #     val_transform,
    #     batch_size=32,
    #     num_workers=processes,
    #     random_split=False,
    #     dataset_class="SpectrogramDir",
    #     save_path_df = dir_save,
    #     discard_line_noise = True,
    #     split_by_inst=False,
    #     binary=True,
    #     previosly_uncompressed=False
    # )

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    # Create model
    n_classes = 3
    models = {
        "CNN_Long_Data": model.CNN_Long_Data(n_classes=n_classes, input_size=input_size, input_length=input_length), # 35,516
        "CNN_Long_Data2": model.CNN_Long_Data2(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_Long_Data3": model.CNN_Long_Data3(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_Long_Data4": model.CNN_Long_Data4(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_Long_Data5": model.CNN_Long_Data5(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_Long_Data6": model.CNN_Long_Data6(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_Resnet": model.custom_resnet34(n_classes = n_classes, input_size=input_size),
        "CNN_RNN_Long_Data1": model.CNN_RNN_Long_Data1(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_RNN_Long_Data2": model.CNN_RNN_Long_Data2(n_classes=n_classes, input_size=input_size, input_length=input_length),
        "CNN_RNN_Long_Data3": model.CNN_RNN_Long_Data3(n_classes=n_classes, input_size=input_size, input_length=input_length),
    }
    model_0 = models[model_name].to(device)
    print(f"Model: {model_0}")
    # Load model if it was trained before
    if os.path.exists(os.path.join(dir_save, "best_model.pt")):
        # load the last checkpoint with the best model
        print(
            f"Previous trained model found in {os.path.join(dir_save, 'best_model.pt')}. Loading... \n",
            flush=True,
        )
        model_0.load_state_dict(torch.load(os.path.join(dir_save, "best_model.pt")))
    elif os.path.exists(os.path.join(dir_save, "checkpoint.pt")):
        # load the last checkpoint with the best model
        print(
            f"Previous trained model found in {os.path.join(dir_save, 'checkpoint.pt')}. Loading... \n",
            flush=True,
        )
        model_0.load_state_dict(torch.load(os.path.join(dir_save, "checkpoint.pt")))
    else:
        print(
            f"No previous model found in {dir_save}.",
        )

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() #weight=cls_weights
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=15)
    #lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model_0
    model_0_results, n_iters = train(
        model=model_0,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler = scheduler,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        dir_save=dir_save,
        validation_freq=validation_freq,
        restore_best_model=False,
        max_iter=max_iter
    )

    # Save results
    path_results = os.path.join(dir_save, "loss.txt")
    with open(path_results, "w") as write_file:
        json.dump(model_0_results, write_file)

    # Restore best model
    model_0.load_state_dict(torch.load(os.path.join(dir_save, "best_model.pt")))
    # Run after training evaluation
    results_val = utils.eval_model(model_0, val_dataloader)
    results_val['n_iters'] = n_iters
    path_results = os.path.join(dir_save, "results_val.txt")
    with open(path_results, "w") as write_file:
        json.dump(results_val, write_file)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument(
        "-p", "--path", type=str, required=True
    )  # Path to save model and results
    parser.add_argument(
        "-c", "--cores", type=int, required=True
    )  # Path to save model and results
    parser.add_argument(
        "-f", "--features", type=str, required=True
    ) 
    parser.add_argument(
        "-s", "--srate", type=int, required=True
    )
    parser.add_argument(
        "-df_val", "--df_val", type=str, required=True
    ) 
    parser.add_argument(
        "-df_train", "--df_train", type=str, required=True
    ) 
    args = parser.parse_args()
    print("test", end="\n", flush=True)
    main(args.model, args.path, int(args.cores), args.features, int(args.srate), args.df_val, args.df_train)
