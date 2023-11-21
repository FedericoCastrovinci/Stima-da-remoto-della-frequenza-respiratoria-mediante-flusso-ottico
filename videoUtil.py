import cv2

import numpy as np
import pandas as pd
import cv2 as cv
from scipy.signal import welch, butter, filtfilt, iirnotch, freqz, stft
from sklearn.decomposition import PCA
from scipy.stats import iqr, median_abs_deviation


def get_fps(videoFileName):
    """
    This method returns the fps of a video file name or path.
    """
    vidcap = cv.VideoCapture(videoFileName)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps
def get_Totfps(videoFileName):
    """
    This method returns the total fps of a video file name or path.
    """
    video = cv2.VideoCapture(videoFileName)
    totFps = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return totFps

def extract_frames_yield(videoFileName):
    """
    Questo metodo produce i fotogrammi di un nome o percorso di file video.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    success, image = vidcap.read()
    
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()
    
def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    Questo metodo viene utilizzato per calcolare gli indici per la creazione di un segnale di finestre sovrapposte.

     Argomenti:
         N (int): lunghezza del segnale.
         wsize (float): dimensione della finestra in secondi.
         passo (float): passo tra finestre sovrapposte in pochi secondi.
         fps (float): fotogrammi al secondo.

     Ritorna:
     Elenco di intervalli, ognuno contiene gli indici di una finestra e un array 1D di tempi in secondi, dove ognuno è il centro di 
     una finestra.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)

class TestResult():
    """ Gestisci i risultati di un test per un determinato set di dati video e più metodi VHR"""
    
    def __init__(self, filename=None):

        if filename == None:
            self.dataFrame = pd.DataFrame()
        else:
            self.dataFrame = pd.read_hdf(filename)
        self.dict = None
        
    def addDataSerie(self):
        # -- store serie
        if self.dict != None:
            #import code; code.interact(local=locals())#nano

            self.dataFrame = self.dataFrame.append(self.dict, ignore_index=True)
            #self.dataFrame = self.dataFrame.concat(self.dict)#problem
            #self.dataFrame=pd.concat([self.dataFrame, self.dict])
    def newDataSerie(self):
        # -- new dict
        D = {}
        D['method'] = ''
        D['dataset'] = ''
        D['videoIdx'] = ''        # video filename
        D['sigFilename'] = ''     # GT signal filename
        D['videoFilename'] = ''   # GT signal filename
        D['EVM'] = False          # True if used, False otherwise
        D['mask'] = ''            # mask used
        D['RMSE'] = ''
        D['MAE'] = ''
        D['PCC'] = ''
        D['MAX'] = ''
        D['telapse'] = ''
        D['bpmGT'] = ''          # GT bpm
        D['bpmES'] = ''
        D['timeGT'] = ''            # GT bpm
        D['timeES'] = ''
        D['processTime'] = ''#mod    
        self.dict = D
    
    def addData(self, key, value):
        self.dict[key] = value
                         
    def saveResults(self, outFilename=None):
        if outFilename == None:
            outFilename = "testResults.h5"
        else:
            self.outFilename = outFilename
        
        # -- save data
        self.dataFrame.to_hdf(outFilename, key='df', mode='w')
        
def BVP_windowing(bvp, wsize, fps, stride=1):
  """ Performs BVP signal windowing

    Args:
      bvp (list/array): full BVP signal
      wsize     (float): size of the window (in seconds)
      fps       (float): frames per seconds
      stride    (float): stride (in seconds)

    Returns:
      bvp_win (list): windowed BVP signal
      timesES (list): times of (centers) windows 
  """
  
  bvp = np.array(bvp).squeeze()
  block_idx, timesES = sliding_straded_win_idx(bvp.shape[0], wsize, stride, fps)
  bvp_win  = []
  for e in block_idx:
      st_frame = int(e[0])
      end_frame = int(e[-1])
      wind_signal = np.copy(bvp[st_frame: end_frame+1])
      bvp_win.append(wind_signal[np.newaxis, :])

  return bvp_win, timesES


class RWsignal:
    """
    Manage (multi-channel, row-wise) respiratory signals, and transforms them in RPMs.
    """
    #nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds

    def __init__(self, data, fs, startTime=0, minHz=0.2, maxHz=0.4, verb=False):
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
        nyquistF = self.fs/2
        fRes = 0.5
        self.nFFT = max(2048, (60*2*nyquistF) / fRes)

    def spectrogram(self, winsize=5):
        """
        Compute the respiratory signal spectrogram restricted to the
        band 6-24 RPM by using winsize (in sec) samples.
        """

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=self.fs*winsize,
                       noverlap=self.fs*(winsize-self.step),
                       boundary='even',
                       nfft=self.nFFT)
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.75 Hz - 4.0 Hz)
        minHz = 0.2
        maxHz = 0.4
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band, :])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in RPM
        self.times = T                     # spectrum times

        # -- RPM estimate by spectrum
        self.rpm = self.freqs[np.argmax(self.spect, axis=0)]

    def displaySpectrum(self, display=False, dims=3):
        """Show the spectrogram of the respiratory signal"""

        # -- check if rpm exists
        try:
            rpm = self.rpm
        except AttributeError:
            self.spectrogram()
            rpm = self.rpm

        t = self.times
        f = self.freqs
        S = self.spect

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=S, x=t, y=f, colorscale="viridis"))
        fig.add_trace(go.Scatter(
            x=t, y=rpm, name='Frequency Domain', line=dict(color='red', width=2)))

        fig.update_layout(autosize=False, height=420, showlegend=True,
                          title='Spectrogram of the respiratory signal',
                          xaxis_title='Time (sec)',
                          yaxis_title='rpm (60*Hz)',
                          legend=dict(
                              x=0,
                              y=1,
                              traceorder="normal",
                              font=dict(
                                family="sans-serif",
                                size=12,
                                color="black"),
                              bgcolor="LightSteelBlue",
                              bordercolor="Black",
                              borderwidth=2)
                          )

        fig.show(renderer=VisualizeParams.renderer)

    def getRPM(self, winsize=5):
        """
        Get the RPM signal extracted from the ground truth respiratory signal.
        """
        self.spectrogram(winsize)
        return self.rpm, self.times

