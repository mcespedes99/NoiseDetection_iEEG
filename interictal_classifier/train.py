import os
import argparse


def main(model_name: str, dir_save: str, processes: int):
    print("start", end="\n", flush=True)
    print("torch", end="\n", flush=True)
    import torch
    from torchvision import transforms
    import sys
    import model
    import data_setup
    from engine import train
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

    # Zip files and transforms
    zip_files = [
        "/home/mcesped/scratch/Datasets/Dataset_Fnusa_np_maps.zip",
        "/home/mcesped/scratch/Datasets/Dataset_Mayo_np_maps.zip",
    ]

    train_transform = transforms.Compose(
        [
            #transforms.Resize((30, 100), antialias=True),
            transforms.RandomHorizontalFlip(p=0.3),
            utils.RandomMask(p=0.4, f_max_size= 3, t_max_size = 10, n_masks_f=2, n_masks_t=2),
        ]
    )
    val_transform = (
        None  # transforms.Compose([transforms.Resize((30, 100), antialias=True)])
    )

    # Create dataloaders
    (
        train_dataloader,
        val_dataloader,
        cls_weights,
    ) = data_setup.create_dataloaders_uncompress(
        zip_files,
        train_transform,
        val_transform,
        batch_size=16,
        num_workers=processes,
        weightedsampler=True,
        random_split=False,
        dataset_class="MultiLevelSpectrogramDir",
    )

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 40

    # Create model
    models = {
        "IeegClassifier": model.IeegClassifier(n_classes=4),
        "IeegClassifier2": model.IeegClassifier2(n_classes=4),
        "IeegClassifier3": model.IeegClassifier3(n_classes=4),
        "MultiLayerClassifier": model.MultiLayerClassifier(n_classes=4),
        "MultiLayerCNN": model.MultiLayerCNN(n_classes=4),
        "MultiLayerCNNSmall": model.MultiLayerCNNSmall(n_classes=4),
        "MultiLayerCNNSmall2": model.MultiLayerCNNSmall2(n_classes=4),
        "MultiLayerCNNTiny": model.MultiLayerCNNTiny(n_classes=4),
        "resnet34": model.custom_resnet34(n_classes=4),
        "SimpleCNN": model.SimpleCNN(n_classes=4),
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
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.0001)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model_0
    model_0_results = train(
        model=model_0,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        dir_save=dir_save,
        restore_best_model=False,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument(
        "-p", "--path", type=str, required=True
    )  # Path to save model and results
    parser.add_argument(
        "-c", "--cores", type=int, required=True
    )  # Path to save model and results
    args = parser.parse_args()
    print("test", end="\n", flush=True)
    main(args.model, args.path, int(args.cores))
