import zipfile
import scipy.io
import pandas as pd
import numpy as np
import os
import re
import shutil
from torchvision import transforms
import custom_dataset
import torch
from features import get_DWT_features, get_CWT_features, get_Hilbert_features, get_CWT_features_bands, get_Hilbert_CWT_features
import scipy.stats as stats
import utils
from pathlib import Path
import argparse


def compute_maps_2048(data, srate=2048):
    import mne
    # Fill info with anything
    info = mne.create_info(['a'], ch_types='seeg', sfreq=srate)
    # Create raw
    raw = mne.io.RawArray(data.reshape(1,-1)/10000, info, verbose='ERROR')
    # Transform 
    transf = transforms.Resize((100, 100), antialias=True)
    # Bandpass filtering with diff bands
    if srate==2048:
        bands = [(20,100), (80,250), (200,600), (500,900), (900, None)]
        images = torch.zeros(6,100,100)
        max_freq = 1000
    elif srate==1200:
        bands = [(20,100), (80,250), (200,None), (500,None)]
        images = torch.zeros(5,100,100)
        max_freq = 590
    # Compute full transform and z-score per freq
    tf, freq = custom_dataset.compute_wavelet_transform(data, srate=srate, min_freq = 1, max_freq=max_freq)
    tf_pytorch = np.expand_dims(tf, axis=0)
    tf_pytorch = transf(torch.from_numpy(tf_pytorch)).squeeze()
    # Z-score freq
    zscore_freq_torch = stats.zscore(tf_pytorch.detach().numpy(), axis=1)
    images[0,:,:] = torch.from_numpy(zscore_freq_torch)
    # Now time wise
    for i, (low_freq, high_freq) in enumerate(bands):
        new_raw = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, verbose='ERROR')
        new_data = new_raw.get_data().squeeze()*10000
        # Compute wavelet transform
        tf, freq = custom_dataset.compute_wavelet_transform(new_data, srate=srate, min_freq = 1, max_freq=max_freq)
        # Convert to tensor to apply tfms
        tf_pytorch = np.expand_dims(tf, axis=0)
        tf_pytorch = transf(torch.from_numpy(tf_pytorch)).squeeze()
        # Z-score time
        zscore_time_torch = stats.zscore(tf_pytorch.detach().numpy(), axis=0)
        images[i+1,:,:] = torch.from_numpy(zscore_time_torch)
    return images

def preprocess_data(signal_array: np.ndarray, df_channel: pd.DataFrame, target_srate:int=2048):
    """
    Uses MNE to downsample the signal. Assumes the signal is 3 seconds long.
    Args:
        signal_array: numpy array containing the signal. In units of V*10-4
        df_channel: DataFrame with the channel info.
    """
    import mne
    # First get sampling rate
    srate = len(signal_array)/3.0
    # Get type and chn name
    chn_name = [df_channel['channel']]
    # To get channel type, map to mne names
    types_map = {
        'depth': 'seeg',
        'strip': 'ecog'
    }
    type_chn = [types_map[df_channel['electrode_type']]]
    # Create info
    info = mne.create_info(chn_name, ch_types=type_chn, sfreq=srate)
    # Create raw
    raw = mne.io.RawArray(signal_array.reshape(1,-1)/10000, info, verbose='ERROR')
    if srate != target_srate:
        # Downsample the signal to target_srate Hz
        raw = raw.copy().resample(target_srate)
    # High pass the signal to 0.5 Hz. Not doing this! 
    # https://sapienlabs.org/lab-talk/pitfalls-of-filtering-the-eeg-signal 
    # raw = raw.copy().filter(l_freq=1, h_freq=None, verbose='ERROR')
    # Visualize
    # filter_params = mne.filter.create_filter(
    #     raw_dn.get_data(), raw_dn.info["sfreq"], l_freq=0.5, h_freq=None
    # )
    # _ = mne.viz.plot_filter(filter_params, raw_dn.info["sfreq"], flim=(0.01, 3))
    return raw.get_data().squeeze()*10000

def convert_to_numpy():
    print('Using 1200 Hz', end='\n', flush=True)
    zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Fnusa.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo.zip']
    # print(zip_files)
    for zip_id, zip_file in enumerate(zip_files):
        with zipfile.ZipFile(zip_file, mode="r") as f:
            # Filename 
            filename = os.path.basename(zip_file)
            # Get all files
            files = f.namelist()
            # Find segments.csv
            reg = re.compile("segments_new.csv")
            seg_path= list(filter(reg.search, files))[0]
            # Get df
            with f.open(seg_path) as myfile:
                df_seg = pd.read_csv(myfile, sep=',', index_col='index')
            # Now downsample every mat file and save to dir
            # Create dir first
            dir_name = filename.split('.')[0]
            new_dir = os.path.join(os.environ.get('SLURM_TMPDIR'), dir_name)
            os.makedirs(new_dir,exist_ok=True)
            # Compute for each mat file
            for idx in df_seg.index.to_numpy():
                # Get the data 
                sid = df_seg.loc[idx, 'segment_id']
                reg = re.compile(sid)
                seg_file = list(filter(reg.search, files))[0]
                with f.open(seg_file) as myfile:
                    data = scipy.io.loadmat(myfile)['data'].squeeze()
                # Preprocess data
                data = preprocess_data(data, df_seg.loc[idx], 1200)
                # Write to tmp file
                np.save(os.path.join(new_dir, sid+'.npy'),data)
            # Also write df to path
            df_seg.to_csv(os.path.join(new_dir, "segments_new.csv"), sep=',', index_label='index')
            #  Create zipfile from dir
            archive_name = os.path.join('/home/mcesped/scratch/Datasets/', dir_name+'_np')
            shutil.make_archive(
                archive_name,
                'zip',
                root_dir=os.environ.get('SLURM_TMPDIR'),
                base_dir=dir_name,
            )
            # Remove files from tmp
            shutil.rmtree(new_dir)

def convert_to_maps_2048(data_type):
    if data_type == 2048:
        print('Maps 2048 Hz', end='\n', flush=True)
        srate=2048
        zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Mayo_2048.zip'] #'/home/mcesped/scratch/Datasets/Dataset_Fnusa_2048.zip'
        out_dir = '/home/mcesped/scratch/Datasets/2048Hz'
    else:
        print('Maps 1200 Hz', end='\n', flush=True)
        srate=1200
        zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Fnusa_np.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo_np.zip']
        out_dir = '/home/mcesped/scratch/Datasets/1200Hz'
    # First find tmp
    tmpdir = utils.get_tmpdir()
    # Now uncompress
    print("\nUncompressing data..", end="\n", flush=True)
    utils.uncompress_zip(zip_files=zip_files, out_dir=tmpdir)
    print("\nDone", end="\n", flush=True)
    # Get dataframe
    df_total = pd.read_csv('/scratch/mcesped/Datasets/segments_mayo_fnusa_curated_big_version.csv', sep=',', index_col='index')
    institutions = ['mayo'] #['fnusa','mayo']
    # New paths
    paths_files = [os.path.join(tmpdir,'mayo/Dataset_Mayo')] #os.path.join(tmpdir,'fnusa/Dataset_Fnusa')
    # print(zip_files)
    zipped_list = list(zip(paths_files, zip_files, institutions))
    for directory_in_str, zip_file, inst in zipped_list:
        print(f"\Computing data.. for {inst}", end="\n", flush=True)
        files = os.listdir(directory_in_str)
        # print(files)
        # Filename 
        filename = os.path.basename(zip_file)
        # Get inst dataframe
        df_seg = df_total.loc[df_total.institution==inst]
        df_seg = df_seg.reset_index(drop=True)
        # Now downsample every mat file and save to dir
        # Create dir first
        dir_name = filename.split('.')[0]
        new_dir = os.path.join(os.environ.get('SLURM_TMPDIR'), dir_name)
        os.makedirs(new_dir, exist_ok=True)
        # Compute for each mat file
        for idx in df_seg.index.to_numpy():
            # Get the data 
            sid = df_seg.loc[idx, 'segment_id']
            # print(sid)
            reg = re.compile(sid)
            seg_file = list(filter(reg.search, files))[0]
            # Open file
            myfile = os.path.join(directory_in_str, seg_file)
            data = np.load(myfile)
            # Preprocess data
            images = compute_maps_2048(data, srate=srate)
            # Write to tmp file
            torch.save(images, os.path.join(new_dir, sid+'.pt'))
        # Also write df to path
        df_seg.to_csv(os.path.join(new_dir, "segments_new.csv"), sep=',', index_label='index')
        #  Create zipfile from dir
        archive_name = os.path.join(out_dir, dir_name+'_maps')
        shutil.make_archive(
            archive_name,
            'zip',
            root_dir=new_dir
        )
        # Remove files from tmp
        shutil.rmtree(new_dir)

def convert_to_maps():
    zip_files = ['/home/mcesped/scratch/Datasets/Dataset_UFlorida_np.zip','/home/mcesped/scratch/Datasets/Dataset_Fnusa_np.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo_np.zip']
    # print(zip_files)
    for zip_id, zip_file in enumerate(zip_files):
        with zipfile.ZipFile(zip_file, mode="r") as f:
            # Filename 
            filename = os.path.basename(zip_file)
            # Get all files
            files = f.namelist()
            # Find segments.csv
            reg = re.compile("segments_new.csv")
            seg_path= list(filter(reg.search, files))[0]
            # Get df
            with f.open(seg_path) as myfile:
                df_seg = pd.read_csv(myfile, sep=',', index_col='index')
            # Now downsample every mat file and save to dir
            # Create dir first
            dir_name = filename.split('.')[0]
            new_dir = os.path.join(os.environ.get('SLURM_TMPDIR'), dir_name)
            os.makedirs(new_dir,exist_ok=True)
            # Compute for each mat file
            for idx in df_seg.index.to_numpy():
                # Get the data 
                sid = df_seg.loc[idx, 'segment_id']
                reg = re.compile(sid)
                seg_file = list(filter(reg.search, files))[0]
                with f.open(seg_file) as myfile:
                    data = np.load(myfile)
                # Preprocess data
                # Compute wavelet transform
                data, freq = custom_dataset.compute_wavelet_transform(data)
                # Convert to maps
                tf = transforms.Resize((30, 100), antialias=True)
                images = custom_dataset.compute_maps_per_band(data, freq, tf)
                # Write to tmp file
                torch.save(images, os.path.join(new_dir, sid+'.pt'))
            # Also write df to path
            df_seg.to_csv(os.path.join(new_dir, "segments_new.csv"), sep=',', index_label='index')
            #  Create zipfile from dir
            archive_name = os.path.join('/home/mcesped/scratch/Datasets/', dir_name+'_maps')
            shutil.make_archive(
                archive_name,
                'zip',
                root_dir=new_dir
            )
            # Remove files from tmp
            shutil.rmtree(new_dir)


def convert_to_DWT():
    import pywt

    zip_files = ['/home/mcesped/scratch/Datasets/Dataset_UFlorida_np.zip','/home/mcesped/scratch/Datasets/Dataset_Fnusa_np.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo_np.zip']
    # print(zip_files)
    for zip_id, zip_file in enumerate(zip_files):
        with zipfile.ZipFile(zip_file, mode="r") as f:
            # Filename 
            filename = os.path.basename(zip_file)
            # Get all files
            files = f.namelist()
            # Find segments.csv
            reg = re.compile("segments_new.csv")
            seg_path= list(filter(reg.search, files))[0]
            # Get df
            with f.open(seg_path) as myfile:
                df_seg = pd.read_csv(myfile, sep=',', index_col='index')
            # Now downsample every mat file and save to dir
            # Create dir first
            dir_name = filename.split('.')[0]
            new_dir = os.path.join(os.environ.get('SLURM_TMPDIR'), dir_name)
            os.makedirs(new_dir,exist_ok=True)
            # Compute for each mat file
            for idx in df_seg.index.to_numpy():
                # Get the data 
                sid = df_seg.loc[idx, 'segment_id']
                reg = re.compile(sid)
                seg_file = list(filter(reg.search, files))[0]
                with f.open(seg_file) as myfile:
                    data = np.load(myfile)
                # Preprocess data
                data = stats.zscore(data)
                # Compute DWT
                list_coeff = pywt.wavedec(data, "db4")
                # Write to tmp file
                np.save(os.path.join(new_dir, sid+'.npy'), data)
            # Also write df to path
            df_seg.to_csv(os.path.join(new_dir, "segments_new.csv"), sep=',', index_label='index')
            #  Create zipfile from dir
            archive_name = os.path.join('/home/mcesped/scratch/Datasets/', dir_name+'_DWTcoeffs')
            shutil.make_archive(
                archive_name,
                'zip',
                root_dir=new_dir
            )
            # Remove files from tmp
            shutil.rmtree(new_dir)

def get_features(data, features: str, srate: int):
    assert features in ['DWT','CWT','Hilbert', 'CWT_Full', 'Combined']
    if features=='DWT':
        data = stats.zscore(data)
        # Compute features through DWT
        data = get_DWT_features(data, "db4")
    elif features == 'CWT':
        data = get_CWT_features_bands(data, srate)
    elif features=='CWT_Full':
        data = get_CWT_features(data, srate)
    elif features=='Hilbert':
        data = get_Hilbert_features(data, srate)
    elif features=='Combined':
        data = get_Hilbert_CWT_features(data, srate)
    return data

def convert_to_features(srate, features: str):
    if srate == 2048:
        print(f'{features} 2048 Hz', end='\n', flush=True)
        zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Fnusa_2048.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo_2048.zip']
        out_dir = '/home/mcesped/scratch/Datasets/2048Hz'
    else:
        print(f'{features} 1200 Hz', end='\n', flush=True)
        zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Fnusa_1200.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo_1200.zip']
        # zip_files = ['/home/mcesped/scratch/Datasets/Dataset_Mayo_np.zip']
        out_dir = '/home/mcesped/scratch/Datasets/1200Hz'
    print("\nUsing zipfiles:", end="\n", flush=True)
    print(zip_files)
    print(f"\nOutputting to directory {out_dir}:", end="\n", flush=True)
    # First find tmp
    tmpdir = utils.get_tmpdir()
    # Now uncompress
    print("\nUncompressing data..", end="\n", flush=True)
    utils.uncompress_zip(zip_files=zip_files, out_dir=tmpdir)
    print("\nDone", end="\n", flush=True)
    # New paths
    paths_files = [os.path.join(tmpdir,'fnusa/Dataset_Fnusa'), os.path.join(tmpdir,'mayo/Dataset_Mayo')]#os.path.join(tmpdir,'fnusa/Dataset_Fnusa')
    # print(zip_files)
    zipped_list = list(zip(paths_files, zip_files))
    for directory_in_str, zip_file in zipped_list:
        print(f"\nComputing data.. for {zip_file}", end="\n", flush=True)
        files = os.listdir(directory_in_str)
        # print(files)
        # Filename 
        filename = os.path.basename(zip_file)
        # Find segments.csv
        myfile = os.path.join(directory_in_str, "segments_new.csv")
        df_seg = pd.read_csv(myfile, sep=',', index_col='index')
        # Now downsample every mat file and save to dir
        # Create dir first
        dir_name = filename.split('.')[0]
        new_dir = os.path.join(os.environ.get('SLURM_TMPDIR'), dir_name)
        os.makedirs(new_dir, exist_ok=True)
        # Compute for each mat file
        for idx in df_seg.index.to_numpy():
            # Get the data 
            sid = df_seg.loc[idx, 'segment_id']
            # print(sid)
            reg = re.compile(sid)
            seg_file = list(filter(reg.search, files))[0]
            # Open file
            myfile = os.path.join(directory_in_str, seg_file)
            data = np.load(myfile)
            # Get features from data
            data = get_features(data, features, srate)
            # Write to tmp file
            np.save(os.path.join(new_dir, sid+'.npy'), data)
        # Also write df to path
        df_seg.to_csv(os.path.join(new_dir, "segments_new.csv"), sep=',', index_label='index')
        #  Create zipfile from dir
        archive_name = os.path.join(out_dir, dir_name+f'_{features}')
        shutil.make_archive(
            archive_name,
            'zip',
            root_dir=new_dir
        )
        # Remove files from tmp
        shutil.rmtree(new_dir)

if __name__=="__main__":
    # Model to use
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--srate", type=int, required=True)
    parser.add_argument("-f", "--features", type=str, required=True)
    args = parser.parse_args()
    convert_to_features(args.srate, args.features)
