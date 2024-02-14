# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import zipfile
import re
import scipy.stats as stats
import torch
from sklearn.model_selection import train_test_split
# from torchvision import transforms
import os
import utils
from features import get_DWT_features, compute_wavelet_transform

# 1. Subclass torch.utils.data.Dataset
class DWTDir(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter.
    def __init__(self, df: pd.DataFrame, input_folder: str) -> None:
        """
        Args:
            df: DataFrame of the same type as segments.csv with only the subjects for the Dataset.
            zip_map: Dict mapping 'institution' column of the dataframe to a zip file.
            transform: transforms.Compose to apply to the data
        """

        self.input_folder = input_folder
        self.df = df
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(df)

    # 3. Overwrite the __len__() method
    def __len__(self):
        return len(self.df)

    # 4. Overwrite the __getitem__() method
    def __getitem__(self, item):
        target = self.df.iloc[item]["category_id"]
        # This applies transforms as well
        data = self.load_data(item)

        return data, target  # return data, label (X, y)

    # 4. Make function to load images
    def load_data(self, index: int) -> np.array:
        "Opens an image via a dataframe and returns it."
        sid = self.df.iloc[index]["segment_id"]
        # Get institution to map to zip file
        inst = self.df.iloc[index]["institution"]
        # Get data
        map_dir = {
            'fnusa': 'Dataset_Fnusa',
            'mayo': 'Dataset_Mayo'
        }
        data = np.load(os.path.join(self.input_folder,f'{inst}/{map_dir[inst]}/{sid}.npy'))
        data = get_DWT_features(data, "db4")
        data = torch.from_numpy(data)
        return data.type(torch.float)


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter.
    def __init__(self, df: pd.DataFrame, zip_files: List[str], transform=None) -> None:
        """
        Args:
            df: DataFrame of the same type as segments.csv with only the subjects for the Dataset.
            zip_map: Dict mapping 'institution' column of the dataframe to a zip file.
            transform: transforms.Compose to apply to the data
        """

        self.zip_files = zip_files
        # Find mapp between institutions and zip files and all the files for each zip
        files = []
        inst_to_zipID = dict()
        for zip_id, zip_file in enumerate(zip_files):
            with zipfile.ZipFile(zip_file, mode="r") as f:
                # Get all files
                files_zip = f.namelist()
                files.append(files_zip)
                # Find segments.csv
                reg = re.compile("segments_new.csv")
                seg_path = list(filter(reg.search, files_zip))[0]
                # Get df
                with f.open(seg_path) as myfile:
                    df_seg = pd.read_csv(myfile, sep=",", index_col="index")
                    # Add the mapping
                    for inst in np.unique(df_seg["institution"]):
                        inst_to_zipID[inst] = zip_id
        # Save vals
        self.files = files
        self.inst_to_zipID = inst_to_zipID
        self.df = df
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(df)

    # 3. Overwrite the __len__() method
    def __len__(self):
        return len(self.df)

    # 4. Overwrite the __getitem__() method
    def __getitem__(self, item):
        target = self.df.iloc[item]["category_id"]
        data = self.load_spectrogram(item)

        # Transform if necessary
        if self.transform:
            data = self.transform(data)

        # Normalize
        data = self.normalize_spectrogram(data.squeeze())

        return data, target  # return data, label (X, y)

    # 4. Make function to load images
    def load_spectrogram(self, index: int) -> np.array:
        "Opens an image via a dataframe and returns it."
        sid = self.df.iloc[index]["segment_id"]
        # Get institution to map to zip file
        zip_file_id = self.inst_to_zipID[self.df.iloc[index]["institution"]]
        # Open zip file to get data
        with zipfile.ZipFile(self.zip_files[zip_file_id], mode="r") as archive:
            files = self.files[zip_file_id]
            reg = re.compile(sid)
            seg_file = list(filter(reg.search, files))[0]
            with archive.open(seg_file) as myfile:
                data = np.load(myfile)
            # print(data.shape)
            # Compute wavelet transform
            data,_ = compute_wavelet_transform(data)
        data = np.expand_dims(data, axis=0)
        # Convert to tensor
        data = torch.from_numpy(data)
        return data.type(torch.float)

    def normalize_spectrogram(self, image: torch.tensor):
        # We need to detach it to compute zscore
        image = stats.zscore(image.detach().numpy(), axis=1)
        # Image could have nans in frequencies with zeros (from augmentation). Replace with zeros
        image[np.isnan(image)] = 0
        # Convert back to tensor
        image = torch.from_numpy(image).type(torch.float)
        image = torch.unsqueeze(image, axis=0)
        return image

class MultiLevelSpectrogram(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter.
    def __init__(self, df: pd.DataFrame, zip_files: List[str], transform=None) -> None:
        """
        Args:
            df: DataFrame of the same type as segments.csv with only the subjects for the Dataset.
            zip_map: Dict mapping 'institution' column of the dataframe to a zip file.
            transform: transforms.Compose to apply to the data
        """

        self.zip_files = zip_files
        # Find mapp between institutions and zip files and all the files for each zip
        files = []
        inst_to_zipID = dict()
        for zip_id, zip_file in enumerate(zip_files):
            with zipfile.ZipFile(zip_file, mode="r") as f:
                # Get all files
                files_zip = f.namelist()
                files.append(files_zip)
                # Find segments.csv
                reg = re.compile("segments_new.csv")
                seg_path = list(filter(reg.search, files_zip))[0]
                # Get df
                with f.open(seg_path) as myfile:
                    df_seg = pd.read_csv(myfile, sep=",", index_col="index")
                    # Add the mapping
                    for inst in np.unique(df_seg["institution"]):
                        inst_to_zipID[inst] = zip_id
        # Save vals
        self.files = files
        self.inst_to_zipID = inst_to_zipID
        self.df = df
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(df)

    # 3. Overwrite the __len__() method
    def __len__(self):
        return len(self.df)

    # 4. Overwrite the __getitem__() method
    def __getitem__(self, item):
        target = self.df.iloc[item]["category_id"]
        # This applies transforms as well
        data = self.load_spectrogram(item)

        return data, target  # return data, label (X, y)

    # 4. Make function to load images
    def load_spectrogram(self, index: int) -> np.array:
        "Opens an image via a dataframe and returns it."
        sid = self.df.iloc[index]["segment_id"]
        # Get institution to map to zip file
        zip_file_id = self.inst_to_zipID[self.df.iloc[index]["institution"]]
        # Open zip file to get data
        with zipfile.ZipFile(self.zip_files[zip_file_id], mode="r") as archive:
            files = self.files[zip_file_id]
            reg = re.compile(sid)
            seg_file = list(filter(reg.search, files))[0]
            with archive.open(seg_file) as myfile:
                data = np.load(myfile)
            # print(data.shape)
        # Compute wavelet transform
        # data, freq = compute_wavelet_transform(data)
        # # Convert to maps
        # images = compute_maps_per_band(data, freq, self.transform)
        return data #images.type(torch.float)

class MultiLevelSpectrogramDir(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter.
    def __init__(self, df: pd.DataFrame, input_folder: str, transform=None) -> None:
        """
        Args:
            df: DataFrame of the same type as segments.csv with only the subjects for the Dataset.
            zip_map: Dict mapping 'institution' column of the dataframe to a zip file.
            transform: transforms.Compose to apply to the data
        """

        self.input_folder = input_folder
        self.df = df
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(df)

    # 3. Overwrite the __len__() method
    def __len__(self):
        return len(self.df)

    # 4. Overwrite the __getitem__() method
    def __getitem__(self, item):
        target = self.df.iloc[item]["category_id"]
        # This applies transforms as well
        data = self.load_image(item)

        # Transform if necessary
        if self.transform:
            data = self.transform(data)

        return data, target  # return data, label (X, y)

    # 4. Make function to load images
    def load_image(self, index: int) -> np.array:
        "Opens an image via a dataframe and returns it."
        sid = self.df.iloc[index]["segment_id"]
        # Get institution to map to zip file
        inst = self.df.iloc[index]["institution"]
        # Get images
        images = torch.load(os.path.join(self.input_folder,f'{inst}/{sid}.pt'))
        images = images[0,:,:]
        images = torch.unsqueeze(images, axis=0)
        return images.type(torch.float)

class SpectrogramDir(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter.
    def __init__(self, df: pd.DataFrame, input_folder: str, features: str, transform=None) -> None:
        """
        Args:
            df: DataFrame of the same type as segments.csv with only the subjects for the Dataset.
            zip_map: Dict mapping 'institution' column of the dataframe to a zip file.
            transform: transforms.Compose to apply to the data
        """

        self.input_folder = input_folder
        self.df = df
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(df)
        assert features in ['CWT', 'Hilbert', 'Combined']
        self.features = features

    # 3. Overwrite the __len__() method
    def __len__(self):
        return len(self.df)

    # 4. Overwrite the __getitem__() method
    def __getitem__(self, item):
        target = self.df.iloc[item]["category_id"]
        # This applies transforms as well
        data = self.load_image(item)

        # Transform if necessary
        if self.transform:
            data = self.transform(data)

        return data, target  # return data, label (X, y)

    # 4. Make function to load images
    def load_image(self, index: int) -> np.array:
        "Opens an image via a dataframe and returns it."
        sid = self.df.iloc[index]["segment_id"]
        # Get institution to map to zip file
        inst = self.df.iloc[index]["institution"]
        # Get images
        images = np.load(os.path.join(self.input_folder,f'{inst}/{sid}.npy'))
        if self.features == 'Hilbert':
            images = torch.from_numpy(images[1,:,:].squeeze())
            images = torch.unsqueeze(images, axis=0)
        elif self.features == 'CWT':
            images = torch.from_numpy(images[0,:,:].squeeze())
            images = torch.unsqueeze(images, axis=0)
        else:
            images = torch.from_numpy(images)
        return images.type(torch.float)

def compute_maps_per_band(image, freq, transform):
    bands = [(20,100), (80,200), (200,400), (400,600)]
    images = torch.zeros(8,30,100)
    for i, (low_freq, high_freq) in enumerate(bands):
        # Get data in this range
        low_freq_id = np.argmin(np.abs(freq-low_freq))
        high_freq_id = np.argmin(np.abs(freq-high_freq))
        tf_band = image[low_freq_id:high_freq_id+1,:]
        # Apply tfms
        if transform:
            # Convert to tensor to apply tfms
            tf_band_tensor = np.expand_dims(tf_band, axis=0)
            tf_band_tensor = transform(torch.from_numpy(tf_band_tensor)).squeeze()
            # Convert back to numpy
            tf_band = tf_band_tensor.detach().numpy()
        # Z-score freq
        zscore_freq_torch = stats.zscore(tf_band, axis=1)
        zscore_freq_torch[np.isnan(zscore_freq_torch)] = 0
        images[i*(2),:,:] = torch.from_numpy(zscore_freq_torch)
        # Z-score time
        zscore_time_torch = stats.zscore(tf_band, axis=0)
        zscore_time_torch[np.isnan(zscore_time_torch)] = 0
        images[i*(2)+1,:,:] = torch.from_numpy(zscore_time_torch)
    return images

def find_classes(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """Finds the classes in the provided dataframe

    Args:
        df (pd.DataFrame): Target dataframe.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # Get the unique classes in the dataframe
    classes = np.unique(df["category_name"]).tolist()
    # Get the associated index
    class_map = dict()
    for unique_class in classes:
        class_idx = df.loc[df["category_name"] == unique_class, "category_id"].iloc[0]
        class_map[unique_class] = class_idx
    return classes, class_map


def get_train_val_sets(
    paths: List[str], val_split: float = 0.3, random_split:bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Gets 2 paths of zip files containing a segments_new.csv and divides the data
    # between training and validation by separating by subject (training and val have diff subjects).
    # assert len(paths) == 2
    # (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.3,random_state=100, stratify=y)
    # Initialize variables
    df_total = []
    files = []
    map_zip_to_inst = dict()
    # Compute total df
    for zip_id, zip_file in enumerate(paths):
        with zipfile.ZipFile(zip_file, mode="r") as f:
            # Get all files
            files_zip = f.namelist()
            files.append(files_zip)  # Save files
            # Find segments.csv
            reg = re.compile("segments_new.csv")
            seg_path = list(filter(reg.search, files_zip))[0]
            # Get df
            with f.open(seg_path) as myfile:
                df = pd.read_csv(myfile, sep=",", index_col="index")
                # Add the mapping
                for inst in np.unique(df["institution"]):
                    map_zip_to_inst[inst] = zip_id
                # Append df
                if len(df_total) == 0:
                    df_total = df
                else:
                    df_total = pd.concat([df_total, df])
    # Get datasets
    df_train, df_val = train_val_split_multiclass(
        df_total, val_split=val_split, margin_allowance=0.02, class_col="institution"
    )
    if random_split:
        df_total = df_total.reset_index(drop=True)
        X = df_total.index.values
        y = df_total["category_id"].values
        (X_train, X_val, _, _) = train_test_split(
            X, y, test_size=0.3, random_state=100, stratify=y
        )
        df_train = df_total.iloc[X_train]
        df_val = df_total.iloc[X_val]
    return df_train, df_val


def train_val_split(df, val_split=0.3, margin_allowance=0.02):
    # List of percentages by subj
    percentages_list = []
    for id_subj in np.unique(df["patient_id"]):
        percentages_list.append(len(df[df["patient_id"] == id_subj]) / len(df))
    # Zip with patients id
    percentages = np.vstack([percentages_list, np.unique(df["patient_id"])])
    val_real = 0
    val_subj = []
    cond = False
    while cond == False:
        # Look for all the percentages lower than val
        mask = percentages[0, :] < val_split - val_real
        percentages_eval = percentages[:, mask]
        if percentages_eval.size != 0:
            # Get the bigger value
            new_val_id = percentages_eval[0, :].argmax()
            if (
                val_split + margin_allowance >= val_real + percentages_eval[0, :].max()
            ):  # %2 margin
                val_real += percentages_eval[0, :].max()
                # add val subj
                val_subj.append(int(percentages_eval[1, new_val_id]))
                # Delete the value from array
                indexes = np.arange(percentages.shape[-1])
                percentages = np.delete(percentages, indexes[mask][new_val_id], axis=-1)
            else:
                cond = True
        else:
            cond = True
    # Get train subj
    train_subj = percentages[1, :].astype(int)
    train_real = 1 - val_real

    return [(train_subj, train_real), (val_subj, val_real)]


def train_val_split_multiclass(
    df, val_split=0.3, margin_allowance=0.02, class_col="institution"
):
    df_val = []
    df_train = []
    for id_class in np.unique(df[class_col]):
        df_class = df[df[class_col] == id_class]
        # Run train val split
        (train_subj, _), (val_subj, _) = train_val_split(
            df_class, val_split, margin_allowance
        )
        # Add train subj df
        for subj in train_subj:
            if len(df_train) == 0:
                df_train = df_class[df_class["patient_id"] == subj]
            else:
                df_train = pd.concat(
                    [df_train, df_class[df_class["patient_id"] == subj]]
                )
        # Add val subj df
        for subj in val_subj:
            if len(df_val) == 0:
                df_val = df_class[df_class["patient_id"] == subj]
            else:
                df_val = pd.concat([df_val, df_class[df_class["patient_id"] == subj]])

    return df_train.reset_index(), df_val.reset_index()