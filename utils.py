from __future__ import division
import torch
import numpy as np
import mediapipe as mp
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal._arraytools import even_ext
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate
from scipy.signal import blackmanharris, fftconvolve
import sys

def Welch_rpm(resp, fps, winsize, minHz=0.1, maxHz=0.4, nfft=2048):
    """
    This method computes the spectrum of a respiratory signal

    Parameters
    ----------
        resp: the respiratory signal
        fps: the fps of the video from which signal is estimated
        winsize: the window size used to compute spectrum
        minHz: the lower bound for accepted frequencies
        maxHz: the upper bound for accepted frequencies
        nfft: the length of fft used

    Returns
    -------
        the array of frequencies and the corrisponding PSD
    """
    step = 1
    nperseg=fps*winsize
    noverlap=fps*(winsize-step)

    # -- periodogram by Welch
    F, P = signal.welch(resp, nperseg=nperseg, noverlap=noverlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()

    Pfreqs = 60*F[band]
    Power = P[:, band]

    return Pfreqs, Power
# modificare maxHz max freq gt / 60
def sig_to_RPM(sig, fps, winsize, nFFT, minHz=0.1, maxHz=0.4):
    #pyVHR/pyVHR/BPM/BPM.py -- BVP_to_BPM()
    rpms = []
    obj = None
    for s in sig:
        rpm_es = simplesig_to_RPM(s, fps, winsize, nFFT, minHz, maxHz)
        rpms.append(rpm_es)
    return rpms

def simplesig_to_RPM(sig, fps, winsize, nFFT, minHz=0.1, maxHz=0.4):
    #pyVHR/pyVHR/BPM/BPM.py -- BVP_to_BPM()
    if sig.shape[0] == 0:
        return np.float32(0.0)
    
    Pfreqs, Power = Welch_rpm(sig, fps, winsize, minHz, maxHz, nFFT)
    
    # -- BPM estimate
    Pmax = np.argmax(Power, axis=1)  # power max
    return Pfreqs[Pmax.squeeze()]

def average_filter(sig, win_length = 5):
    """
    This method applies to a signal an average filter

    Parameters
    ----------
        sig: the respiratory signal
        win_length: the length of the window used to apply the average filter

    Returns
    -------
        the filtered signal
    """
    res = []
    sig = even_ext(np.array(sig), win_length, axis=-1)
    for i in np.arange(win_length, len(sig)-win_length+1):
        window = np.sum(sig[i-win_length:i+win_length])
        res.append(1/(1+2*win_length)*window)
    return res

def butter_lowpass_filter(data, cutoff, fs, order=6):
    """
    This method applies to a signal a butter lowpass filter

    Parameters
    ----------
        data: the respiratory signal
        cutoff: the cutoff frequency
        fs: the sampling frequency
        order: the order of the filter

    Returns
    -------
        the filtered signal
    """
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def plot_mask(mask):
    """
    This method plots the mask given as input

    Parameters
    ----------
        mask: the input mask

    Returns
    -------
        the plotted mask
    """
    plt.imshow(mask, interpolation='nearest')
    plt.show()

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)



def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, currently has trouble with finding the true peak

    """
    # Calculate autocorrelation and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[int(len(corr)/2):]

    # Find the first low point
    d = diff(corr)
    start, = np.nonzero(np.ravel(d > 0))
    start = start[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable, due to peaks that occur between samples.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def freq_from_crossings(sig, fs):
    """Estimatcorr[len(corr)/2:]e frequency by counting zero crossings

    Pros: Fast, accurate (increasing with data length).  Works well for long low-noise sines, square, triangle, etc.

    Cons: Doesn't work if there are multiple zero crossings per cycle, low-frequency baseline shift, noise, etc.

    """
    # Find all indices right before a rising-edge zero crossing
    indices, = np.nonzero(np.ravel((sig[1:] >= 0) & (sig[:-1] < 0)))

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    #crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better. Spline, cubic, whatever

    return fs / average(diff(crossings))

def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT

    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000003 Hz for 1000 Hz, for instance).  Due to parabolic interpolation
    being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with data length

    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.  Better method would try to identify the fundamental

    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(abs(f), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def snr(sig, fs, nperseg, noverlap):
    """
    This method computes the SNR of a signal

    Parameters
    ----------
        sig: the respiratory signal
        fs: the sampling frequency
        nperseg: the length of each segment
        noverlap: the number of points to overlap between segments

    Returns
    -------
        the SNR of the given signal
    """
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    num = 0
    den = 0
    for i in np.arange(len(freqs)):
        if freqs[i]>=0.1 and freqs[i]<=0.4:
            num+=psd[i]
        if freqs[i]>=0 and freqs[i]<=4:
            den+=psd[i]
    if den!=0:
        return num/den
    else:

        return -1

def pad_rgb_signal(sig, fps, win_size):
    """
    This method applies padding to a windowed rgb signal

    Parameters
    ----------
        sig: the respiratory signal
        fps: the sampling frequency
        win_size: the length of each segment

    Returns
    -------
        The padded RGB respiratory signal
    """
    sig = np.swapaxes(sig,0,1)

    nperseg = fps * win_size

    new_sig = []
    for roi in sig:
        red = [frame[0] for frame in roi]
        green = [frame[1] for frame in roi]
        blue = [frame[2] for frame in roi]

        red = even_ext(np.asarray(red), int(nperseg//2), axis=-1)
        green = even_ext(np.asarray(green), int(nperseg//2), axis=-1)
        blue = even_ext(np.asarray(blue), int(nperseg//2), axis=-1)

        new_roi = []
        for i in np.arange(len(red)):
            new_roi.append([red[i], green[i], blue[i]])

        new_sig.append(new_roi)


    return np.swapaxes(new_sig,0,1)

def get_channel(sig, channel):
    """
    This method select from a windowed rgb signal a single channel

    Parameters
    ----------
        sig: the respiratory signal
        channel: the channel index (0:red, 1:green, 2:blue)

    Returns
    -------
        The signal resukting from the selection
    """
    res = []
    for win in sig:
        row = []
        for roi in win:
            row.append(roi[channel])
        res.append(row)
    return res

def bland_altman_plot(data1, data2, *args, **kwargs):
    # data1 ground truth
    # data2 estimation
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')