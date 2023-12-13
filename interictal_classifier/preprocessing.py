import zipfile
import scipy.io
import pandas as pd
import numpy as np
import os
import re
import shutil

def preprocess_data(signal_array: np.ndarray, df_channel: pd.DataFrame):
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
    if srate != 1200:
        # Downsample the signal to 1200 Hz
        raw = raw.copy().resample(1200)
    # High pass the signal to 0.5 Hz. Not doing this! 
    # https://sapienlabs.org/lab-talk/pitfalls-of-filtering-the-eeg-signal 
    # raw = raw.copy().filter(l_freq=1, h_freq=None, verbose='ERROR')
    # Visualize
    # filter_params = mne.filter.create_filter(
    #     raw_dn.get_data(), raw_dn.info["sfreq"], l_freq=0.5, h_freq=None
    # )
    # _ = mne.viz.plot_filter(filter_params, raw_dn.info["sfreq"], flim=(0.01, 3))
    return raw.get_data().squeeze()*10000

def main():
    zip_files = ['/home/mcesped/scratch/Datasets/Dataset_UFlorida.zip','/home/mcesped/scratch/Datasets/Dataset_Fnusa.zip', '/home/mcesped/scratch/Datasets/Dataset_Mayo.zip']
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
                data = preprocess_data(data, df_seg.loc[idx])
                # Write to tmp file
                np.save(os.path.join(new_dir, sid+'.npy'),data)
            # Also write df to path
            df_seg.to_csv(os.path.join(new_dir, "segments_new.csv"), sep=',')
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

if __name__=="__main__":
    main()