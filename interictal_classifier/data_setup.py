# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter.
    def __init__(self, df: pd.DataFrame, zip_map: Dict[str, str], transform=None) -> None:
        """
        Args:
            df: DataFrame of the same type as segments.csv with only the subjects for the Dataset.
            zip_map: Dict mapping 'institution' column of the dataframe to a zip file.
            transform: transforms.Compose to apply to the data
        """
        
        self.zip_map = zip_map
        self.df = df
        self.transform = transform
        self.NFFF = 200
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        sid = self.df.iloc[item]["segment_id"]
        target = self.df.iloc[item]["category_id"]
        # Get institution to map to zip file
        zip_file = zip_map[self.df.iloc[item]["institution"]]
        # Open zip file to get data
        with 
            data = sio.loadmat(self.path + "{}".format(sid))["data"]
            _, _, data = signal.spectrogram(
                data[0, :], fs=5000, nperseg=256, noverlap=128, nfft=1024
            )

        data = data[: self.NFFF, :]
        data = stats.zscore(data, axis=1)
        data = np.expand_dims(data, axis=0)
        return data, target

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a dataframe and returns it."
        seg_id = self.df.iloc[index]["segment_id"]
        # Open zip file to get data
        with 
            data = sio.loadmat(self.path + "{}".format(sid))["data"]
            _, _, data = signal.spectrogram(
                data[0, :], fs=5000, nperseg=256, noverlap=128, nfft=1024
            )
        return data

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)

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
    classes = np.unique(df['category_name']).tolist()
    # Get the associated index
    class_map = dict()
    for unique_class in classes:
        class_idx = df.loc[df['category_name']==unique_class, 'category'][0]
        class_map[unique_class] = class_idx
    return classes, class_map

def train_val_split(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Gets 2 paths of zip files containing a segments_new.csv and divides the data
    # between training and validation by separating by subject (training and val have diff subjects).
    assert len(paths) == 2
