import os

from torchvision import transforms
from torch.utils.data import DataLoader
import custom_dataset
import torch
import numpy as np
import utils


def create_dataloaders(
    zip_paths: str,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    batch_size: int,
    num_workers: int = None,
    weightedsampler: bool = True,
    random_split: bool = False,
    dataset_class: str = "ImageFolderCustom",
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    if not num_workers:
        num_workers = os.cpu_count()
    # Create train val sets
    df_train, df_val = custom_dataset.get_train_val_sets(
        zip_paths, val_split=0.3, random_split=random_split
    )
    # Shuffle them
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)

    # Deal with class imbalance
    # Train set
    train_labels = df_train.category_id.to_numpy()
    # Probability per class
    class_sample_counts = np.unique(train_labels, return_counts=True)[1]
    cls_weights = 1 / torch.Tensor(class_sample_counts)
    # Map back to elements
    weights = cls_weights[train_labels]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(train_labels), replacement=False
    )

    # Get dataset class
    dataset_dict = {
        "ImageFolderCustom": custom_dataset.ImageFolderCustom,
        "MultiLevelSpectrogram": custom_dataset.MultiLevelSpectrogram,
        "MultiLevelSpectrogramDir": custom_dataset.MultiLevelSpectrogramDir,
    }
    dataset_custom = dataset_dict[dataset_class]
    if dataset_class == "MultiLevelSpectrogramDir":
        # Create datasets
        print("\nCreating dataset folders...\n", end="\n", flush=True)
        train_folder = custom_dataset.create_dataset(
            df_train, zip_paths, "train", processes=num_workers
        )
        val_folder = custom_dataset.create_dataset(
            df_train, zip_paths, "validation", processes=num_workers
        )
        # Create dataset class
        print("\nCreating dataset classes...\n", end="\n", flush=True)
        train_data = dataset_custom(
            df=df_train, input_folder=train_folder, transform=transform_train
        )
        val_data = dataset_custom(
            df=df_val, input_folder=val_folder, transform=transform_val
        )
    else:
        # Use ImageFolderCustom to create dataset(s)
        train_data = dataset_custom(
            df=df_train, zip_files=zip_paths, transform=transform_train
        )
        val_data = dataset_custom(
            df=df_val, zip_files=zip_paths, transform=transform_val
        )

    # Get class names
    class_names = train_data.classes

    # Wrapper to support full batches
    if batch_size == -1:
        print("Full")
        train_batch_size = len(train_data)
        val_batch_size = len(val_data)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size

    # Turn images into data loaders
    if weightedsampler:
        train_dataloader = DataLoader(
            train_data,
            batch_size=train_batch_size,
            # shuffle=True, # Don't work with sampler
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            train_data,
            batch_size=train_batch_size,
            shuffle=True,  # Don't work with sampler
            num_workers=num_workers,
            pin_memory=True,
        )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, cls_weights


def create_dataloaders_uncompress(
    zip_paths: str,
    df_train_path: str,
    df_val_path: str,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    batch_size: int,
    features: str,
    num_workers: int = None,
    dataset_class: str = "ImageFolderCustom",
    previosly_uncompressed: bool = False,
    binary:bool = False,
):
    # Created dataloaders given val and train dfs
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    import pandas as pd
    if not num_workers:
        num_workers = os.cpu_count()
    # Read given dataframe
    df_train = pd.read_csv(df_train_path, sep=",", index_col="index")
    df_val = pd.read_csv(df_val_path, sep=",", index_col="index")
    
    # Binary classification. 0=Noise, 1=Others (physiological and pathological)
    if binary:
        print('Grouping Pathology and physiology in one class\n', end='\n', flush=True)
        df_train = df_train.replace({'category_id': {2: 0, 1:0, 0:1}})
        df_val = df_val.replace({'category_id': {2: 0, 1:0, 0:1}})
   
    # Shuffle them
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)

    # Deal with class imbalance
    # Train set
    train_labels = df_train.category_id.to_numpy()
    print(train_labels)
    # Probability per class
    class_sample_counts = np.unique(train_labels, return_counts=True)[1]
    cls_weights = 1 / torch.Tensor(class_sample_counts)
    print(cls_weights)
    # Map back to elements
    weights = cls_weights[train_labels]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(train_labels), replacement=True
    )

    # Get dataset class
    dataset_custom = custom_dataset.SpectrogramDir
    # 1. Uncompress data
    # First find tmp
    tmpdir = utils.get_tmpdir()
    if not previosly_uncompressed:
        # Now uncompress
        print("\nUncompressing data..", end="\n", flush=True)
        utils.uncompress_zip(zip_files=zip_paths, out_dir=tmpdir)
    # 2. Create dataset class
    print("\nCreating dataset classes...", end="\n", flush=True)
    train_data = dataset_custom(
        df=df_train, input_folder=tmpdir, features=features, transform=transform_train
    )
    val_data = dataset_custom(
        df=df_val, input_folder=tmpdir, features=features, transform=transform_val
    )
    
    # Get class names
    class_names = train_data.classes

    # Wrapper to support full batches
    if batch_size == -1:
        print("Full")
        train_batch_size = len(train_data)
        val_batch_size = len(val_data)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size

    # Turn images into data loaders
    # if binary:
    #     print('using sampler')
    #     train_dataloader = DataLoader(
    #         train_data,
    #         batch_size=train_batch_size,
    #         # shuffle=True,  # Don't work with sampler
    #         sampler=sampler,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #     )
    # else:
    train_dataloader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,  # Don't work with sampler
        # sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, class_names
def create_dataloaders_manual(
    zip_paths: str,
    df_path: str,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    batch_size: int,
    num_workers: int = None,
    random_split: bool = False,
    dataset_class: str = "ImageFolderCustom",
    save_path_df: str = None,
    previosly_uncompressed: bool = False,
    discard_line_noise: bool = False,
    split_by_inst: bool = False,
    binary:bool = False
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    import pandas as pd
    if not num_workers:
        num_workers = os.cpu_count()
    # Read given dataframe
    df = pd.read_csv(df_path, sep=",", index_col="index")
    if split_by_inst:
        print("\nSplitting by inst...", end="\n", flush=True)
        df_train = df.loc[df.institution=='fnusa']
        df_test_val = df.loc[df.institution=='mayo']
        df_test, df_val = custom_dataset.train_val_split_multiclass(
                df_test_val, val_split=0.5, margin_allowance=0.02, class_col="institution"
            )
    else:
        print("\nSplitting by subj...", end="\n", flush=True)
        # Create train val sets
        df_train, df_test_val = custom_dataset.train_val_split_multiclass(
            df, val_split=0.4, margin_allowance=0.02, class_col="institution"
        )
        df_test, df_val = custom_dataset.train_val_split_multiclass(
                df_test_val, val_split=0.5, margin_allowance=0.02, class_col="institution"
            )
    # Save them if required
    if save_path_df:
        for df_tmp, df_name in [(df_train, 'df_train.csv'), (df_val, 'df_val.csv'), (df_test, 'df_test.csv')]:
            df_tmp.to_csv(os.path.join(save_path_df, df_name), sep=",", index_label='index')

    # Discard powerline noise if required
    if discard_line_noise:
        # Remove class
        df_train = df_train.loc[df_train.category_id != 0]
        df_val = df_val.loc[df_val.category_id != 0]
        # Update ids to start in zero
        df_train.loc[:,'category_id'] = df_train.category_id - 1
        df_val.loc[:,'category_id'] = df_val.category_id - 1
    
    # Binary classification. 0=Noise, 1=Others (physiological and pathological)
    if binary:
        print('Grouping Pathology and physiology in one class\n', end='\n', flush=True)
        df_train = df_train.replace({'category_id': {2: 0, 1:0, 0:1}})
        df_val = df_val.replace({'category_id': {2: 0, 1:0, 0:1}})
   
    # Shuffle them
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)

    # Deal with class imbalance
    # Train set
    train_labels = df_train.category_id.to_numpy()
    # Probability per class
    class_sample_counts = np.unique(train_labels, return_counts=True)[1]
    cls_weights = 1 / torch.Tensor(class_sample_counts)
    print(cls_weights)
    # Map back to elements
    weights = cls_weights[train_labels]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(train_labels), replacement=True
    )

    # Get dataset class
    dataset_dict = {
        "MultiLevelSpectrogramDir": custom_dataset.MultiLevelSpectrogramDir,
        "SpectrogramDir": custom_dataset.SpectrogramDir
    }
    dataset_custom = dataset_dict[dataset_class]
    # 1. Uncompress data
    # First find tmp
    tmpdir = utils.get_tmpdir()
    if not previosly_uncompressed:
        # Now uncompress
        print("\nUncompressing data..", end="\n", flush=True)
        utils.uncompress_zip(zip_files=zip_paths, out_dir=tmpdir)
    # 2. Create dataset class
    print("\nCreating dataset classes...", end="\n", flush=True)
    train_data = dataset_custom(
        df=df_train, input_folder=tmpdir, transform=transform_train
    )
    val_data = dataset_custom(
        df=df_val, input_folder=tmpdir, transform=transform_val
    )
    
    # Get class names
    class_names = train_data.classes

    # Wrapper to support full batches
    if batch_size == -1:
        print("Full")
        train_batch_size = len(train_data)
        val_batch_size = len(val_data)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size

    # Turn images into data loaders
    if binary:
        print('using sampler')
        train_dataloader = DataLoader(
            train_data,
            batch_size=train_batch_size,
            # shuffle=True,  # Don't work with sampler
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            train_data,
            batch_size=train_batch_size,
            shuffle=True,  # Don't work with sampler
            # sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, class_names



def create_dataloaders_tree(
    zip_paths: str,
    df_path: str,
    batch_size: int,
    num_workers: int = None,
    random_split: bool = False,
    previosly_uncompressed: bool = False,
    discard_line_noise:bool = True
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    import pandas as pd
    if not num_workers:
        num_workers = os.cpu_count()
    # Read given dataframe
    df = pd.read_csv(df_path, sep=",", index_col="index")
    # Create train val sets
    # Create train val sets
    df_train, df_test_val = custom_dataset.train_val_split_multiclass(
        df, val_split=0.4, margin_allowance=0.02, class_col="institution"
    )
    df_test, df_val = custom_dataset.train_val_split_multiclass(
            df_test_val, val_split=0.5, margin_allowance=0.02, class_col="institution"
        )
    # Shuffle them
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    # Discard powerline noise if required
    if discard_line_noise:
        # Remove class
        df_train = df_train.loc[df_train.category_id != 0]
        df_val = df_val.loc[df_val.category_id != 0]
        # Update ids to start in zero
        df_train.loc[:,'category_id'] = df_train.category_id - 1
        df_val.loc[:,'category_id'] = df_val.category_id - 1
    print('Grouping Pathology and physiology in one class\n', end='\n', flush=True)
    df_train = df_train.replace({'category_id': {2: 0, 1:0, 0:1}})
    df_val = df_val.replace({'category_id': {2: 0, 1:0, 0:1}})

    # 1. Uncompress data
    # First find tmp
    tmpdir = utils.get_tmpdir()
    if not previosly_uncompressed:
        # Now uncompress
        print("\nUncompressing data..", end="\n", flush=True)
        utils.uncompress_zip(zip_files=zip_paths, out_dir=tmpdir)
        # 2. Create dataset class
        print("\nCreating dataset classes...", end="\n", flush=True)

    # Use ImageFolderCustom to create dataset(s)
    train_data = custom_dataset.Dataset_DWT(df=df_train, input_folder=tmpdir)
    val_data = custom_dataset.Dataset_DWT(df=df_val, input_folder=tmpdir)

    # Get class names
    class_names = train_data.classes

    # Wrapper to support full batches
    if batch_size == -1:
        print("Full")
        train_batch_size = len(train_data)
        val_batch_size = len(val_data)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,  # Don't work with sampler
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, class_names
