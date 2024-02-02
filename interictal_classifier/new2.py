import os

def main():
    print('start', end="\n", flush=True)
    print('torch', end="\n", flush=True)
    import torch
    from torchvision import transforms
    import sys
    import model
    import data_setup
    from engine import train
    import torch.nn as nn
    print('finish import', end="\n", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    # stdout_fileno = sys.stdout
    # stdout_fileno.write('test\n')
    print('test', end="\n", flush=True)

    # Zip files and transforms
    zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Fnusa_np.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo_np.zip']

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((100, 100), antialias=True)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((100, 100), antialias=True)
    ])

    # Create dataloaders
    train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(zip_files, train_transform, val_transform, batch_size=32, num_workers=2)

    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    # Recreate an instance of TinyVGG
    model_0 = model.custom_resnet34(4).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_0 
    model_0_results = train(model=model_0, 
                            train_dataloader=train_dataloader,
                            test_dataloader=val_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn, 
                            epochs=NUM_EPOCHS,
                            device=device)

    # End the timer and stdout_fileno.write out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds", end="\n", flush=True)

    # Save the model
    # save_model(model=model_0,
    #            target_dir="models",
    #            model_name="05_going_modular_cell_mode_tinyvgg_model.pth")

if __name__ == "__main__":
    print('test', end="\n", flush=True)
    main()