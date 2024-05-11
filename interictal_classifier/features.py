import scipy
import numpy as np
from collections import defaultdict, Counter
import pywt

# Code partially recovered from: https://github.com/taspinar/siml
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    kurtosis_value = scipy.stats.kurtosis(list_values)
    skewness_value = scipy.stats.skew(list_values)
    return [n5, n25, n75, n95, median, mean, std, var, rms, kurtosis_value, skewness_value]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(
        np.diff(np.array(list_values) > np.nanmean(list_values))
    )[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


def get_DWT_features(signal, waveletname):
    import scipy.stats as stats
    features = []
    signal = stats.zscore(signal)
    list_coeff = pywt.wavedec(signal, waveletname)
    # print(len(list_coeff))
    for coeff in list_coeff:
        features += get_features(coeff)
    return np.array(features)

def get_CWT_features(
    data_sig: np.ndarray,
    srate: int = 1200,
):
    # Bandpass filtering with diff bands
    if srate==2048:
        max_freq = 1000
    elif srate==1200:
        max_freq = 590
    # Compute full transform and z-score per freq
    time_freq_data, freq = compute_wavelet_transform(data_sig, srate=srate, min_freq = 1, max_freq=max_freq)
    return time_freq_data

def compute_wavelet_transform(
    data_sig: np.ndarray,
    srate: int = 1200,
    min_freq: int = 1,
    max_freq: int = 590,
    freq_step: int = 5,
):
    import scipy.fft as scifft
    import scipy.stats as stats

    # variable number of wavelet cycles
    # setup parameters
    time = np.arange(
        -1, 1 + 1 / srate, 1 / srate
    )  # best practice is to have time=0 at the center of the wavelet
    frex = np.arange(min_freq, max_freq, freq_step)  # frequency of wavelet, in Hz

    half_wave = int((len(time) - 1) / 2)

    # FFT parameters

    nKern = len(time)

    nData = len(data_sig)

    nConv = nKern + nData - 1

    dataX = scifft.fft(data_sig, nConv)

    # initialize output time-frequency data

    tf = np.zeros((len(frex), nData))

    for fi in range(len(frex)):
        # create wavelet and get its FFT
        # nCycles: determined experientally to compensate for the time/freq trade-off
        nCycles = 0.0000363292 * frex[fi] ** 2 + 0.215155 * frex[fi] + 2.23622
        s = nCycles / (2 * np.pi * frex[fi])
        cmw = np.multiply(
            np.exp(2 * 1j * np.pi * frex[fi] * time),
            np.exp(-(time**2) / (2 * s**2)),
        )

        cmwX = scifft.fft(cmw, nConv)

        # max-value normalize the spectrum of the wavelet
        cmwX = cmwX / np.max(cmwX)

        # run convolution
        conv = scifft.ifft(np.multiply(cmwX, dataX), nConv)
        conv = conv[half_wave:-half_wave]

        # put power data into big matrix
        tf[fi, :] = np.abs(conv) ** 2

    return tf, frex


def get_CWT_features_bands(
    data_sig: np.ndarray,
    srate: int = 1200,
):
    import scipy.fft as scifft
    import scipy.stats as stats
    import mne

    if srate==2048:
        # params: tuples of (freq, nCycles)
        # Based on bands: (20,100), (80,250), (200,600), (500,900)
        params = [(60, 4), (165,5), (395,5), (700,8)]
        low_pass = 900
        n_remove = 25
        output = np.zeros([5,len(data_sig)-2*n_remove])
    elif srate==1200:
        # Based on bands [(20,100), (80,250), (200,590), (500,590)]
        params = [(60, 4), (165,5), (395,4.5), (545,28)]
        low_pass = None
        n_remove = 15
        output = np.zeros([5,len(data_sig)-2*n_remove])
    elif srate==1024:
        # Based on bands [(20,100), (80,250), (200,500), (450,510)]
        n_remove = 13
        params = [(60, 3), (165,5), (350,5), (480,35)]
        low_pass = None
        output = np.zeros([5,len(data_sig)-2*n_remove])

    # First row: loss pass filtered data with 
    filtered_data = data_sig
    if low_pass:   
        # Fill info with anything
        info = mne.create_info(['chn'], ch_types='seeg', sfreq=srate)
        # Create raw
        raw = mne.io.RawArray(data_sig.reshape(1,-1)/10000, info, verbose='ERROR')
        # Filter
        new_raw = raw.copy().filter(l_freq=None, h_freq=low_pass, verbose='ERROR', method='iir', iir_params={'order':3, 'ftype':'butter'})
        filtered_data = new_raw.get_data().squeeze()*10000
    # Removing first and last 0.01 s elements due to edge effects
    output[0,:] = stats.zscore(filtered_data)[n_remove:-n_remove]

    # variable number of wavelet cycles
    # setup parameters
    time = np.arange(
        -1, 1 + 1 / srate, 1 / srate
    )  # best practice is to have time=0 at the center of the wavelet
    half_wave = int((len(time) - 1) / 2)

    # FFT parameters

    nKern = len(time)

    nData = len(data_sig)

    nConv = nKern + nData - 1

    dataX = scifft.fft(data_sig, nConv)

    for fi, (freq, nCycles) in enumerate(params):
        # create wavelet and get its FFT
        # nCycles: determined experientally to compensate for the time/freq trade-off
        s = nCycles / (2 * np.pi * freq)
        cmw = np.multiply(
            np.exp(2 * 1j * np.pi * freq * time),
            np.exp(-(time**2) / (2 * s**2)),
        )

        cmwX = scifft.fft(cmw, nConv)

        # max-value normalize the spectrum of the wavelet
        cmwX = cmwX / np.max(cmwX)

        # run convolution
        conv = scifft.ifft(np.multiply(cmwX, dataX), nConv)
        conv = conv[half_wave:-half_wave]

        # put power data into big matrix
        output[fi+1, :] = stats.zscore(np.abs(conv) ** 2)[n_remove:-n_remove]

    return output

def get_Hilbert_features(
    data_sig: np.ndarray,
    srate: int = 1200,
):
    import scipy.fft as scifft
    import scipy.stats as stats
    from scipy.signal import hilbert
    import mne

    if srate==2048:
        # params: tuples of (freq, nCycles)
        # Based on bands: (20,100), (80,250), (200,600), (500,900)
        n_remove = 25
        bands = [(20,100), (80,250), (200,600), (500,900)]
        low_pass = 900
        output = np.zeros([5,len(data_sig)-2*n_remove])
    elif srate==1200:
        # Based on bands [(20,100), (80,250), (200,590), (500,590)]
        n_remove = 15
        bands = [(20,100), (80,250), (200,590), (500,590)]
        low_pass = None
        output = np.zeros([5,len(data_sig)-2*n_remove])
    elif srate==1024:
        # Based on bands [(20,100), (80,250), (200,590), (500,590)]
        n_remove = 13
        bands = [(20,100), (80,250), (200,500), (450,510)]
        low_pass = None
        output = np.zeros([5,len(data_sig)-2*n_remove])

    # Convert to MNE
    # Fill info with anything
    info = mne.create_info(['chn'], ch_types='seeg', sfreq=srate)
    # Create raw
    raw = mne.io.RawArray(data_sig.reshape(1,-1)/10000, info, verbose='ERROR')
    # First row: loss pass filtered data with 
    filtered_data = data_sig
    if low_pass:   
        # Filter
        new_raw = raw.copy().filter(l_freq=None, h_freq=low_pass, verbose='ERROR', method='iir', iir_params={'order':3, 'ftype':'butter'})
        filtered_data = new_raw.get_data().squeeze()*10000
    output[0,:] = stats.zscore(filtered_data)[n_remove:-n_remove]

    # Run hilbert transform
    for fi, (low_freq, high_freq) in enumerate(bands):
        # Bandpass filter the data
        new_raw = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, verbose='ERROR', method='iir', iir_params={'order':3, 'ftype':'butter'})
        filtered_data = new_raw.get_data().squeeze()*10000
        # Compute hilbert transform
        analytic_signal = hilbert(filtered_data)
        # Compute envelope as squared abs value and save
        output[fi+1, :] = stats.zscore(np.square(np.abs(analytic_signal)))[n_remove:-n_remove]

    return output

def get_Hilbert_CWT_features(
    data_sig: np.ndarray,
    srate: int = 1200,
):
    if srate==2048:
        n_remove = 25
    elif srate==1200:
        n_remove = 15
    elif srate==1024:
        n_remove = 13
    # Get CWT features
    data_CWT = get_CWT_features_bands(data_sig, srate)
    # Get Hilbert features
    data_Hilbert = get_Hilbert_features(data_sig, srate)

    # Combine them
    output = np.zeros([2, 5, len(data_sig)-2*n_remove])
    
    output[0,:,:] = data_CWT
    output[1,:,:] = data_Hilbert
    
    return output
