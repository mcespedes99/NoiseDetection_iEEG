# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import zipfile
import re
import scipy
import scipy.signal as signal
import scipy.stats as stats
import torch

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
                seg_path= list(filter(reg.search, files_zip))[0]
                # Get df
                with f.open(seg_path) as myfile:
                    df_seg = pd.read_csv(myfile, sep=',', index_col='index')
                    # Add the mapping
                    for inst in  np.unique(df_seg['institution']):
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
        # Probably need to convert to tf image

        # Transform if necessary
        if self.transform:
            return self.transform(data), target  # return data, label (X, y)
        else:
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
            print(data.shape)
            # Compute wavelet transform
            data = compute_wavelet_transform(data)
            # _, _, data = signal.spectrogram(
            #     data, fs=1200, nperseg=256, noverlap=128, nfft=1024
            # )
        data = stats.zscore(data, axis=1)
        data = np.expand_dims(data, axis=0)
        # Convert to tensor
        data = torch.from_numpy(data)
        return data


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
        class_idx = df.loc[df['category_name']==unique_class, 'category_id'].iloc[0]
        class_map[unique_class] = class_idx
    return classes, class_map

def get_train_val_sets(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Gets 2 paths of zip files containing a segments_new.csv and divides the data
    # between training and validation by separating by subject (training and val have diff subjects).
    assert len(paths) == 2
    # Initialize variables
    df_total = []
    files = []
    map_zip_to_inst = dict()
    # Compute total df
    for zip_id, zip_file in enumerate(paths):
        with zipfile.ZipFile(zip_file, mode="r") as f:
            # Get all files
            files_zip = f.namelist()
            files.append(files_zip) #Save files
            # Find segments.csv
            reg = re.compile("segments_new.csv")
            seg_path= list(filter(reg.search, files_zip))[0]
            # Get df
            with f.open(seg_path) as myfile:
                df = pd.read_csv(myfile, sep=',', index_col='index')
                # Add the mapping
                for inst in  np.unique(df['institution']):
                    map_zip_to_inst[inst] = zip_id
                # Append df
                if len(df_total)==0:
                    df_total = df
                else:
                    df_total = pd.concat([df_total, df])
    # Get datasets
    df_train, df_val = train_val_split_multiclass(df_total, val_split=0.3, margin_allowance = 0.02, class_col='institution')

    return df_train, df_val

def train_val_split(df, val_split=0.3, margin_allowance = 0.02):
    # List of percentages by subj
    percentages_list = []
    for id_subj in np.unique(df['patient_id']):
        percentages_list.append(len(df[df['patient_id']==id_subj])/len(df))
    # Zip with patients id
    percentages = np.vstack([percentages_list, np.unique(df['patient_id'])])
    val_real = 0
    val_subj = []
    cond = False
    while cond == False:
        # Look for all the percentages lower than val
        mask = percentages[0,:]<val_split-val_real
        percentages_eval = percentages[:,mask]
        if percentages_eval.size!=0:
            # Get the bigger value 
            new_val_id = percentages_eval[0,:].argmax()
            if val_split+margin_allowance >= val_real+percentages_eval[0,:].max(): #%2 margin
                val_real += percentages_eval[0,:].max()
                # add val subj
                val_subj.append(int(percentages_eval[1,new_val_id]))
                # Delete the value from array
                indexes = np.arange(percentages.shape[-1])
                percentages = np.delete(percentages, indexes[mask][new_val_id], axis=-1)
            else:
                cond = True
        else:
            cond = True
    # Get train subj
    train_subj = percentages[1,:].astype(int)
    train_real = 1-val_real
    
    return [(train_subj, train_real), (val_subj, val_real)]

def train_val_split_multiclass(df, val_split=0.3, margin_allowance = 0.02, class_col='institution'):
    df_val = []
    df_train = []
    for id_class in np.unique(df[class_col]):
        df_class = df[df[class_col]==id_class]
        # Run train val split
        (train_subj, _), (val_subj, _) = train_val_split(df_class, val_split, margin_allowance)
        # Add train subj df
        for subj in train_subj:
            if len(df_train)==0:
                df_train = df_class[df_class['patient_id']==subj]
            else:
                df_train = pd.concat([df_train, df_class[df_class['patient_id']==subj]])
        # Add val subj df
        for subj in val_subj:
            if len(df_val)==0:
                df_val = df_class[df_class['patient_id']==subj]
            else:
                df_val = pd.concat([df_val, df_class[df_class['patient_id']==subj]])
    
    return df_train.reset_index(), df_val.reset_index()

def compute_wavelet_transform(data_sig: np.ndarray,
                              srate: int = 1200,
                              min_freq: int = 1,
                              max_freq: int = 590,
                              freq_step: int = 5):
    import scipy.fft as scifft
    import scipy.stats as stats

    # variable number of wavelet cycles
    # setup parameters
    time  = np.arange(-1,1+1/srate, 1/srate) # best practice is to have time=0 at the center of the wavelet
    frex  = np.arange(min_freq,max_freq,freq_step)        # frequency of wavelet, in Hz

    half_wave = int((len(time)-1)/2)

    # FFT parameters

    nKern = len(time)

    nData = len(data_sig)

    nConv = nKern+nData-1

    dataX = scifft.fft(data_sig, nConv)

    # initialize output time-frequency data

    tf = np.zeros((len(frex),nData))

    for fi in range(len(frex)):
        
        # create wavelet and get its FFT
        # nCycles: determined experientally to compensate for the time/freq trade-off
        nCycles = 0.0000363292 * frex[fi]**2 + 0.215155*frex[fi] + 2.23622 
        s = nCycles/(2*np.pi*frex[fi])
        cmw = np.multiply(np.exp(2*1j*np.pi*frex[fi]*time), np.exp(-time**2/(2*s**2)))
        
        cmwX = scifft.fft(cmw, nConv)

        # max-value normalize the spectrum of the wavelet
        cmwX = cmwX/np.max(cmwX)
        
        # run convolution
        conv = scifft.ifft(np.multiply(cmwX,dataX),nConv)
        conv = conv[half_wave:-half_wave]
        
        # put power data into big matrix
        tf[fi,:] = np.abs(conv)**2
    
    return tf