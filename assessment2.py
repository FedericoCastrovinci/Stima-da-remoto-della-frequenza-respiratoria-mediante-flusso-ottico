import h5py
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as ss
import scikit_posthocs as sp
import math
from autorank import autorank, plot_stats, create_report, latex_table
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase, Colorbar
import itertools
import seaborn as sns

# from pyVHR.analysis.pipeline import TestResult
import time


from utils import *
from algorithms2 import *
from videoUtil import *
from errorUtil import *


def get_errors(rpm, gt_rpm, times, gt_times):
    """
    A partire dall'rpm stimato e dal relativo time array, calcola le metriche di errore e correlazione rispetto alla verità fondamentale
     Argomenti:
         giri/min:
             - Il segnale dei giri
         gt_rpm:
             - Il segnale rpm della verità di base
         volte:
             - l'array temporale relativo ai giri stimati<
         gt_times:
             - l'array temporale relativo al numero di giri al suolo
     Ritorna:
         Le metriche di errore e correlazione
    """
    RMSE = RMSEerror(np.expand_dims(rpm, axis=0), gt_rpm, times, gt_times)
    MAE = MAEerror(np.expand_dims(rpm, axis=0), gt_rpm, times, gt_times)
    MAX = MAXError(np.expand_dims(rpm, axis=0), gt_rpm, times, gt_times)
    PCC = PearsonCorr(np.expand_dims(rpm, axis=0), gt_rpm, times, gt_times)
    CCC = LinCorr(np.expand_dims(rpm, axis=0), gt_rpm, times, gt_times)

    return RMSE, MAE, MAX, PCC, CCC


def compute_lin_errors(path, dataset, of_type, evm=False):
    res = TestResult()
    i = 0

    available_datasets = ["cohface", "bp4d", "cohface_EVM_mov"]
    assert dataset in available_datasets, "\nDataset not recognized!!"
    # pc portatile
    # res.saveResults("C:\\Users\\fede2\\OneDrive\\Documenti\\università\\tirocinio\\Tesi_Castrovinci-main\\"+dataset+"_res_lin_2.h5")
    # pc fisso
    # res.saveResults("C:\\Users\\Federico\Downloads\\wetransfer_tesi_castrovinci-main\\Tesi_Castrovinci-main\\Results\\"+dataset+"_res_lin_"+of_type+".h5")
    # server
    res.saveResults("Results/" + dataset + "_res_lin_" + of_type + ".h5")
    for dir1 in os.listdir(path):
        # if dir1 == "Physiology" or dir1 == "BP4D_sample_accepted.txt":
        #    continue
        if dir1 != "MTTS_CAN_preds":
            x = "/" + dir1
            # if dir1!="36": break

            # for dir2 in os.listdir(path+dir1+"/"):
            for dir2 in os.listdir(path + x):
                # ...\cohface\dir1
                win_size = 27  # cohface-based parameter

                # ESTIMATED RESPIRATORY SIGNAL

                if dataset == "cohface" or dataset == "cohface_EVM_mov":
                    fps_gt = 32
                    minF = 0.1
                    maxF = 0.4
                    #maxF = 0.4 default value
                    # video_path = path+dir1+"/"+dir2+"/"+"data.avi"
                    video_path = (
                        path + "/" + str(dir1) + "/" + str(dir2) + "/" + "data.avi"
                    )
                    # print(video_path)#debug

                if dataset == "bp4d":
                    fps_gt = 1000
                    minF = 0
                    maxF = 200 / 60
                    # path_gt = path+dir1+"/"+dir2+"/"
                    path_gt = path + "/" + str(dir1) + "/" + str(dir2) + "/"
                    accepted = open("BP4D_accepted.txt").read().splitlines()
                    video_path = (
                        path + "/" + str(dir1) + "/" + str(dir2) + "/" + "vid.avi"
                    )
                    # path+ dir1 +"/"+dir2+"/"+"vid.avi"
                    if dir1+"/"+dir2+"/" not in accepted:
                        continue

                fps = get_fps(video_path)

                nyquistF = fps / 2
                fRes = 0.5
                nFFT = max(2048, (60 * 2 * nyquistF) / fRes)
                print("calcolo lin su video ", video_path[len(video_path) - 20 :])
                if dataset == "cohface_EVM_mov":
                    start = time.process_time()
                    sig = lin_EVM(video_path, dataset=dataset)
                else:
                    start = time.process_time()
                    sig = lin(video_path, dataset=dataset, of_type=of_type)
                end = time.process_time()
                nperseg = fps * win_size

                # pad signal to get a RPM estimate for each second
                sig = even_ext(np.array(sig), int(nperseg // 2), axis=-1)

                # win_size: dimensione della finestra (in secondi)-->posso mettere tutto il video
                sig_win, times = BVP_windowing(sig, win_size, fps, stride=1)

                # rearrange time array
                times = times - times[0]
                #sig_to_RPM modificare maxF
                rpm = sig_to_RPM(sig_win, fps, win_size, nFFT, minHz=minF, maxHz=maxF)
                rpm = signal.medfilt(rpm, kernel_size=3)

                # GROUND TRUTH
                if dataset == "cohface" or dataset == "cohface_EVM_mov":
                    f = h5py.File(path + "/" + str(dir1) + "/" + str(dir2) + "/" + "data.hdf5","r",)
                    gt = np.array(f["respiration"])
                    gt = gt[np.arange(0, len(gt), 8)]

                if dataset == "bp4d":
                    gt = np.loadtxt(path_gt + "Resp_Volts.txt")
                    # cambio il filtraggio Wn=[0.1, 0.5] default value
                    b, a = butter(N=2, Wn=[0.1, 0.75], fs=fps_gt, btype="bandpass")
                    y = filtfilt(b, a, gt)
                    gt = (2 * (y - np.min(y))) / (np.max(y) - np.min(y)) - 1

                gt_sig = RWsignal(gt, fps_gt, minHz=minF, maxHz=maxF)
                gt_rpm, gt_times = gt_sig.getRPM(winsize=win_size)

                gt_rpm = signal.medfilt(gt_rpm, kernel_size=3)

                # remove padding
                padding = int(win_size // 2)

                gt_times = gt_times[padding : -padding - 1]
                gt_times -= gt_times[0]

                times = times[padding:-padding]
                times -= times[0]

                rpm = rpm[padding:-padding]
                gt_rpm = gt_rpm[padding : -padding - 1]

                # Computes various error/quality measures
                RMSE, MAE, MAX, PCC, CCC = get_errors(rpm, gt_rpm, times, gt_times)
                processTime = (end - start) / get_Totfps(video_path)
                SNR = snr(sig=sig, fs=fps_gt, nperseg=win_size, noverlap=1)
                # SNR = get_SNR(sig_win, fps_gt, gt_sig, times)
                print("RMSE", RMSE)  # debug
                print("MAE", MAE)  # debug
                print("MAX", MAX)  # debug
                print("PCC", PCC)  # debug
                print("CCC", CCC)  # debug
                print("SNR", SNR)  # debug
                print("processTime", processTime)  # debug

                # -- save results
                res.newDataSerie()
                res.addData("dataset", dataset)
                res.addData("method", "Lin")
                res.addData("videoIdx", i)
                res.addData("RMSE", RMSE)
                res.addData("MAE", MAE)
                res.addData("MAX", MAX)
                res.addData("PCC", PCC)
                res.addData("CCC", CCC)
                res.addData("sig", sig)
                res.addData("rpmGT", gt_rpm)
                res.addData("rpmES", rpm)
                res.addData("timeGT", gt_times)
                res.addData("timeES", times)
                res.addData("SNR", SNR)
                res.addData("processTime", processTime)
                res.addData("videoFilename",path + "/" + str(dir1) + "/" + str(dir2) + "/data.avi")
                # res.addData('videoFilename', video_path)
                res.addDataSerie()

                i += 1

    # res.saveResults("C:\\Users\\Federico\Downloads\\wetransfer_tesi_castrovinci-main\\Tesi_Castrovinci-main\\Results\\"+dataset+"_res_lin_"+of_type+".h5")
    res.saveResults("Results/" + dataset + "_res_lin_" + of_type + ".h5")
    return res

def compute_lin_errors_sig(path, dataset, of_type, evm=False):
    res = TestResult()

    available_datasets = ["cohface", "bp4d"]
    assert dataset in available_datasets, "\nDataset not recognized!!"
    # pc portatile
    # res.saveResults("C:\\Users\\fede2\\OneDrive\\Documenti\\università\\tirocinio\\Tesi_Castrovinci-main\\"+dataset+"_res_lin_2.h5")
    # server
    df = pd.read_hdf("Results\\" + dataset + "_res_lin_" + of_type + ".h5")
    res.saveResults("Results_sig\\" + dataset + "_res_lin_" + of_type + ".h5")
    
        
    win_size = 27  # cohface-based parameter

    # ESTIMATED RESPIRATORY SIGNAL

    if dataset == "cohface" or dataset == "cohface_EVM_mov":
        fps_gt = 32
        minF = 0.1
        maxF = 0.5
        #maxF = 0.4 default value
        fps=20
        

    if dataset == "bp4d":
        fps_gt = 1000
        minF = 0
        maxF = 200 / 55
        fps=25
        #maxF = 200 / 60 default value
    #fps = get_fps(video_path)
    
    nyquistF = fps / 2
    fRes = 0.5
    nFFT = max(2048, (60 * 2 * nyquistF) / fRes)
    df.dropna(inplace=True)
    
    nperseg = fps * win_size
    #164 per cohface 112 per bp4d
    for i in range(112):
        sig = average_filter(df["sig"][i])
        gt_rpm = df["rpmGT"][i]
        gt_times = df["timeGT"][i]
        # pad signal to get a RPM estimate for each second
        #sig = even_ext(np.array(sig), int(nperseg // 2), axis=-1)
        # win_size: dimensione della finestra (in secondi)-->posso mettere tutto il video
        sig_win, times = BVP_windowing(sig, win_size, fps, stride=1)
        # rearrange time array
        times = times - times[0]
        #sig_to_RPM modificare maxF
        rpm = sig_to_RPM(sig_win, fps, win_size, nFFT, minHz=minF, maxHz=maxF)
        rpm = signal.medfilt(rpm, kernel_size=3)
        i+=1

        # remove padding
        padding = int(win_size // 2)
        times = times[padding:-padding]
        times -= times[0]
        rpm = rpm[padding:-padding]

        # Computes various error/quality measures
        RMSE, MAE, MAX, PCC, CCC = get_errors(rpm, gt_rpm, times, gt_times)    

        res.newDataSerie()
        res.addData("dataset", dataset)
        res.addData("method", "Lin")
        res.addData("videoIdx", i)
        res.addData("RMSE", RMSE)
        res.addData("MAE", MAE)
        res.addData("MAX", MAX)
        res.addData("PCC", PCC)
        res.addData("CCC", CCC)
        res.addData("sig", sig)
        res.addData("rpmGT", gt_rpm)
        res.addData("rpmES", rpm)
        res.addData("timeGT", gt_times)
        res.addData("timeES", times)
        i+=1
        # res.saveResults("C:\\Users\\Federico\Downloads\\wetransfer_tesi_castrovinci-main\\Tesi_Castrovinci-main\\Results\\"+dataset+"_res_lin_"+of_type+".h5")
        res.addDataSerie()
    res.addData("processTime", df["processTime"])
    res.addDataSerie()
    res.saveResults("Results_sig\\" + dataset + "_res_lin_" + of_type + ".h5")
    return 


class StatAnalysisTest:
    """
    Statistic analyses for multiple datasets and multiple rPPG methods
    """

    def __init__(self, filepath, join_data=False, remove_outliers=False):
        """
        Argomenti:
             percorso del file:
                 - Il percorso del file contenente i risultati da testare
             join_data:
                 - 'True' - Se filepath è una cartella, unire i dataframe contenuti nella cartella (da utilizzare quando si desidera unire più risultati dalla stessa pipeline sullo stesso set di dati)
                 - 'False' - (predefinito) Da utilizzare se si desidera testare la stessa pipeline (eventualmente con più metodi) su più set di dati
             remove_outlier:
                 - 'True' - Rimuove i valori anomali dai dati prima del test statistico
                 - 'False' - (predefinito) nessuna rimozione di valori anomali
        """

        if os.path.isdir(filepath):
            self.multidataset = True
            self.path = filepath + "/"
            self.datasetsList = os.listdir(filepath)
        elif os.path.isfile(filepath):
            self.multidataset = False
            self.datasetsList = [filepath]
            self.path = ""
        else:
            raise "Error: filepath is wrong!"

        self.join_data = join_data
        self.available_metrics = ["MAE", "RMSE", "PCC", "CCC", "SNR"]
        self.remove_outliers = remove_outliers

        # -- get data
        self.__getMethods()
        self.metricSort = {
            "MAE": "min",
            "RMSE": "min",
            "PCC": "max",
            "CCC": "max",
            "SNR": "max",
        }
        self.scale = {
            "MAE": "log",
            "RMSE": "log",
            "PCC": "linear",
            "CCC": "linear",
            "SNR": "linear",
        }

        self.use_stats_pipeline = False

    def __any_equal(self, mylist):
        equal = []
        for a, b in itertools.combinations(mylist, 2):
            equal.append(a == b)
        return np.any(equal)

    def run_stats(
        self, methods=None, metric="CCC", approach="frequentist", print_report=True
    ):
        """
        Esegue la procedura di test statistico selezionando automaticamente il test appropriato per i dati disponibili.
         Argomenti:
             metodi:
                 - I metodi rPPG da analizzare
             metrico:
                 - 'MAE' - Errore assoluto medio
                 - 'RMSE' - Errore quadratico medio della radice
                 - 'PCC' - Coefficiente di correlazione di Pearson
                 - 'CCC' - Coefficiente di correlazione della concordanza
                 - 'SNR' - Rapporto segnale/rumore
             approccio:
                 - 'frequentist' - (predefinito) Utilizza i test di ipotesi frequentista per l'analisi
                 - 'bayesian' - Utilizza i test di ipotesi bayesiana per l'analisi
             stampa il resoconto:
                 - 'Vero' - (predefinito) stampa un report della procedura di verifica dell'ipotesi
                 - 'False' - Non stampa alcun rapporto
         Ritorna:
             Y_df: un DataFrame panda contenente i dati su cui è stata eseguita l'analisi statistica
             fig: una figura matplotlib che mostra il risultato dell'analisi statistica (una figura vuota se è stato scelto il test di Wilcoxon)
        """
        metric = metric.upper()
        assert metric in self.available_metrics, "Error! Available metrics are " + str(
            self.available_metrics
        )

        # -- Method(s)
        if methods is not None:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
            else:
                self.methods = methods

        assert (
            approach == "frequentist" or approach == "bayesian"
        ), "Approach should be 'frequentist' or bayesian, not " + str(approach)

        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]

        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()
        self.ndataset = Y.shape[0]

        if metric == "MAE" or metric == "RMSE":
            order = "ascending"
        else:
            order = "descending"
        m_names = ["_Farneback", "_Lucaskanade_dense", "_RLOF"]
        # m_names = [x.upper().replace('CUPY_', '').replace('CPU_', '').replace('TORCH_', '') for x in self.methods] #['LIN']
        if self.__any_equal(m_names):
            m_names = self.methods
        Y_df = pd.DataFrame(Y, columns=m_names)

        results = autorank(
            Y_df, alpha=0.05, order=order, verbose=False, approach=approach
        )
        self.stat_result = results
        self.use_stats_pipeline = True

        if approach == "bayesian":
            res_df = results.rankdf.iloc[:, [0, 1, 4, 5, 8]]
            print(res_df)

        if print_report:
            print(" ")
            create_report(results)
            print(" ")

        fig = plt.figure(figsize=(12, 5))
        fig.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
        _, ax = self.computeCD(approach=approach, ax=ax)

        return Y_df, fig

    def SignificancePlot(self, methods=None, metric="MAE"):
        """
        Restituisce un grafico di significatività dei risultati del test di ipotesi
        """

        # -- Method(s)
        if methods == None:
            methods = self.methods
        else:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
            else:
                self.methods = methods

        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]

        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()

        # -- Significance plot, a heatmap of p values
        methodNames = [
            x.upper().replace("CUPY_", "").replace("CPU_", "").replace("TORCH_", "")
            for x in self.methods
        ]
        if self.__any_equal(methodNames):
            methodNames = self.methods
        Ypd = pd.DataFrame(Y, columns=methodNames)
        ph = sp.posthoc_nemenyi_friedman(Ypd)
        cmap = ["1", "#fb6a4a", "#08306b", "#4292c6", "#c6dbef"]
        heatmap_args = {
            "cmap": cmap,
            "linewidths": 0.25,
            "linecolor": "0.5",
            "clip_on": False,
            "square": True,
            "cbar_ax_bbox": [0.85, 0.35, 0.04, 0.3],
        }

        fig = plt.figure(figsize=(10, 7))
        ax, cbar = sp.sign_plot(ph, cbar=True, **heatmap_args)
        ax.set_title("p-vals")
        return fig

    def computeCD(
        self,
        ax=None,
        avranks=None,
        numDatasets=None,
        alpha="0.05",
        display=True,
        approach="frequentist",
    ):
        """
        Restituisce la differenza critica e il diagramma della differenza critica per il test post-hoc di Nemenyi se è stato scelto l'approccio frequentista
        Restituisce un grafico dei risultati del test di significatività bayesiana altrimenti
        """
        cd = self.stat_result.cd
        if display and approach == "frequentist":
            stats_fig = plot_stats(self.stat_result, allow_insignificant=True, ax=ax)
        elif display and approach == "bayesian":
            stats_fig = self.plot_bayesian_res(self.stat_result)
        return cd, stats_fig

    def plot_bayesian_res(self, stat_result):
        """
        Rappresenta graficamente i risultati del test di significatività bayesiana
        """
        dm = stat_result.decision_matrix.copy()
        cmap = ["1", "#fb6a4a", "#08306b", "#4292c6"]  # , '#c6dbef']
        heatmap_args = {
            "cmap": cmap,
            "linewidths": 0.25,
            "linecolor": "0.5",
            "clip_on": False,
            "square": True,
            "cbar_ax_bbox": [0.85, 0.35, 0.04, 0.3],
        }
        dm[dm == "inconclusive"] = 0
        dm[dm == "smaller"] = 1
        dm[dm == "larger"] = 2
        np.fill_diagonal(dm.values, -1)

        pl, ax = plt.subplots()
        ax.imshow(dm.values.astype(int), cmap=ListedColormap(cmap))
        labels = list(dm.columns)
        # Major ticks
        ax.set_xticks(np.arange(0, len(labels), 1))
        ax.set_yticks(np.arange(0, len(labels), 1))
        # Labels for major ticks
        ax.set_xticklabels(labels, rotation="vertical")
        ax.set_yticklabels(labels)
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=1)
        ax.set_title("Metric: " + self.metric)
        cbar_ax = ax.figure.add_axes([0.85, 0.35, 0.04, 0.3])
        cbar = ColorbarBase(
            cbar_ax, cmap=ListedColormap(cmap), boundaries=[0, 1, 2, 3, 4]
        )
        cbar.set_ticks(np.linspace(0.5, 3.5, 4))
        cbar.set_ticklabels(["None", "equivalent", "smaller", "larger"])
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor("0.5")
        cbar.ax.tick_params(size=0)
        return pl

    def displayBoxPlot(self, methods=None, metric="MAE", scale=None, title=True):
        """
        Mostra la distribuzione della popolazione con box-plot
        """
        metric = metric.upper()

        # -- Method(s)
        if methods is None:
            methods = self.methods
        else:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
            else:
                self.methods = methods

        if sorted(list(set(self.methods))) != sorted(self.methods):
            methods = self.datasetsList
        else:
            methods = self.methods

        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]
        if scale == None:
            scale = self.scale[metric]

        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()

        # -- display box plot
        fig = self.boxPlot(methods, metric, Y, scale=scale, title=title)
        return fig

    def displayResult(self, methods=None, metric="MAE", scale=None, title=True):
        """
        Mostra la distribuzione della popolazione con box-plot
        """
        metric = metric.upper()

        # -- Method(s)
        if methods is None:
            methods = self.methods
        else:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
            else:
                self.methods = methods

        if sorted(list(set(self.methods))) != sorted(self.methods):
            methods = self.datasetsList
        else:
            methods = self.methods

        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]
        if scale == None:
            scale = self.scale[metric]

        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()

        # -- display box plot
        somma = 0
        for val in Y:
            somma += val
        return somma / len(Y)

    def boxPlot(self, methods, metric, Y, scale, title):
        """
        Crea il box plot
        """

        #  Y = mat(n-datasets,k-methods)

        k = len(methods)

        if not (k == Y.shape[1]):
            raise ("error!")

        offset = 50
        fig = go.Figure()

        methodNames = [
            x.upper().replace("CUPY_", "").replace("CPU_", "").replace("TORCH_", "")
            for x in methods
        ]
        if self.__any_equal(methodNames):
            methodNames = methods
        for i in range(k):
            yd = Y[:, i]
            name = methodNames[i]
            if len(np.argwhere(np.isnan(yd)).flatten()) != 0:
                print(
                    f"Warning! Video {self.dataFrame[0]['videoFilename'][np.argwhere(np.isnan(yd)).flatten()[0]]} contains NaN value for method {name}"
                )
                print("Number of NaN values", len(np.argwhere(np.isnan(yd)).flatten()))
                yd = yd[~np.isnan(yd)]
                # continue
            # -- set color for box
            if (
                metric == "MAE"
                or metric == "RMSE"
                or metric == "TIME_REQUIREMENT"
                or metric == "SNR"
            ):
                med = np.median(yd)
                col = str(min(200, 5 * int(med) + offset))
            if metric == "CC" or metric == "PCC" or metric == "CCC":
                med = 1 - np.abs(np.median(yd))
                col = str(int(200 * med) + offset)

            # -- add box
            fig.add_trace(
                go.Box(
                    y=yd,
                    name=name,
                    boxpoints="all",
                    jitter=0.7,
                    # whiskerwidth=0.2,
                    fillcolor="rgba(" + col + "," + col + "," + col + ",0.5)",
                    line_color="rgba(0,0,255,0.5)",
                    marker_size=2,
                    line_width=2,
                )
            )

        gwidth = np.max(Y) / 10

        if title:
            tit = "Metric: " + metric
            top = 40
        else:
            tit = ""
            top = 10

        fig.update_layout(
            title=tit,
            yaxis_type=scale,
            xaxis_type="category",
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                # dtick=gwidth,
                gridcolor="rgb(255,255,255)",
                gridwidth=0.1,
                zerolinewidth=2,
                titlefont=dict(size=30),
            ),
            font=dict(family="monospace", size=16, color="rgb(20,20,20)"),
            margin=dict(
                l=20,
                r=10,
                b=20,
                t=top,
            ),
            paper_bgcolor="rgb(250, 250, 250)",
            plot_bgcolor="rgb(243, 243, 243)",
            showlegend=False,
        )

        # fig.show()
        return fig

    def saveStatsData(self, methods=None, metric="MAE", outfilename="statsData.csv"):
        """
        Salva le statistiche dei dati su disco
        """
        Y = self.getStatsData(methods=methods, metric=metric, printTable=False)
        np.savetxt(outfilename, Y)

    def getStatsData(self, methods=None, metric="MAE", printTable=True):
        """
        Calcola le statistiche dei dati
        """
        # -- Method(s)
        if methods == None:
            methods = self.methods
        else:
            if set(methods) <= set(self.methods):
                raise ("Some method is wrong!")
            else:
                self.methods = methods

        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]

        # -- get data from dataset(s)
        #    return Y = mat(n-datasets,k-methods)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()

        # -- add median and IQR
        I = ss.iqr(Y, axis=0)
        M = np.median(Y, axis=0)
        Y = np.vstack((Y, M))
        Y = np.vstack((Y, I))

        methodNames = [x.upper() for x in self.methods]
        dataseNames = self.datasetNames
        dataseNames.append("Median")
        dataseNames.append("IQR")
        df = pd.DataFrame(Y, columns=methodNames, index=dataseNames)
        if printTable:
            display(df)

        return Y, df

    def __remove_outliers(self, df, factor=3.5):
        """
        Rimuove gli outlier. Un punto dati è considerato un valore anomalo se
        si trova al di fuori del fattore per l'intervallo interquartile della distribuzione dei dati
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_out = df[
            ~((df < (Q1 - factor * IQR)) | (df > (Q3 + factor * IQR))).any(axis=1)
        ]
        return df_out

    def __getDataMono(self):
        mag = self.mag
        metric = self.metric
        methods = self.methods

        frame = self.dataFrame[0]

        # -- loop on methods
        Y = []

        for method in methods:
            vals = frame[frame["method"] == method][metric]
            if mag == "min":
                data = [v[np.argmin(v)] for v in vals]
            else:
                data = [v[np.argmax(v)] for v in vals]
            Y.append(data)

        if self.remove_outliers:
            res = pd.DataFrame(np.array(Y).T)
            res = self.__remove_outliers(res)
            res = res.to_numpy()
        else:
            res = np.array(Y).T

        return res

    def __getData(self):
        mag = self.mag
        metric = self.metric
        methods = self.methods

        # -- loop on datasets
        Y = []
        m_list = []
        for i, frame in enumerate(self.dataFrame):
            # -- loop on methods
            y = []
            for method in methods:
                vals = frame[frame["method"] == method][metric]
                if vals.empty:
                    continue
                m_list.append(method)
                if mag == "min":
                    data = [v[np.argmin(v)] for v in vals]
                else:
                    data = [v[np.argmax(v)] for v in vals]
                y.append(data)
            y = np.array(y)

            if not self.join_data:
                Y.append(np.mean(y, axis=1))
            else:
                Y.append(y.T)

        if not self.join_data:
            res = np.array(Y)
        else:
            self.methods = m_list
            n_dpoints = [curr_y.shape[0] for curr_y in Y]
            if len(set(n_dpoints)) != 1:
                raise (
                    "There should be the exact same number of elements in each dataset to join when 'join_data=True'"
                )
            res = np.hstack(Y)

        if self.remove_outliers:
            res = pd.DataFrame(res)
            res = self.__remove_outliers(res)
            res = res.to_numpy()

        return res

    def __getMethods(self):
        mets = []
        dataFrame = []
        N = len(self.datasetsList)

        # -- load dataframes
        self.datasetNames = []
        for file in self.datasetsList:
            filename = self.path + file
            self.datasetNames.append(file)
            data = pd.read_hdf(filename)

            mets.append(set(list(data["method"])))
            dataFrame.append(data)

        if not self.join_data:
            # -- method names intersection among datasets
            methods = set(mets[0])
            if N > 1:
                for m in range(1, N - 1):
                    methods.intersection(mets[m])
            methods = list(methods)
        else:
            methods = sum([list(m) for m in mets], [])
            if sorted(list(set(methods))) != sorted(methods):
                raise (
                    "Found multiple methods with the same name... Please ensure using different names for each method when 'join_data=True'"
                )

        methods.sort()
        self.methods = methods
        self.dataFrame = dataFrame


def comparison_table(database, type, value):
    plt.figure(figsize=(12, 4))
    riga = []
    data = []
    i = 0
    for method in type:
        df = pd.read_hdf("Results\\" + database + "_res_lin_" + method + ".h5", key="df")
        # mae e rmse
        for metric in value:
            ris = float(np.nanmean(df[metric].to_numpy()))
            riga.append(round(ris, 4))
        data.append(riga)
        riga = []
    plt.axis('off')
    plt.table(cellText=data,rowLabels=type, colLabels=value, loc="center")

    plt.show()

def comparison_table_sig(database, type, value):
    plt.figure(figsize=(12, 4))
    riga = []
    data = []
    i = 0
    for method in type:
        df = pd.read_hdf("Results_sig\\" + database + "_res_lin_" + method + ".h5", key="df")
        # mae e rmse
        for metric in value:
            ris = float(np.nanmean(df[metric].to_numpy()))
            riga.append(round(ris, 4))
        data.append(riga)
        riga = []
    plt.axis('off')
    plt.table(cellText=data,rowLabels=type, colLabels=value, loc="center")

    plt.show()

def get_meanData(df,x):
    list=[]
    for num in df[x]:
        ris = float(np.nanmean(num))
        list.append(ris)
    return list


def bland_altman(database, type):
    rpmGT=[]
    rpmES=[]
    a=1
    plt.figure(figsize=(16,10))
    for method in type:
        plt.subplot(3,3,a)
        df = pd.read_hdf("Results\\"+database+"_res_lin_"+method+".h5", key="df")
        rpmGT= get_meanData(df,x="rpmGT")
        rpmES= get_meanData(df,x="rpmES")
        bland_altman_plot(rpmGT,rpmES)
        rpmGT=[]
        rpmES=[]
        plt.title(method)
        a+=1
    plt.show()

def bland_altman_sig(database, type):
    rpmGT=[]
    rpmES=[]
    a=1
    plt.figure(figsize=(16,10))
    for method in type:
        plt.subplot(3,3,a)
        df = pd.read_hdf("Results_sig\\"+database+"_res_lin_"+method+".h5", key="df")
        rpmGT= get_meanData(df,x="rpmGT")
        rpmES= get_meanData(df,x="rpmES")
        bland_altman_plot(rpmGT,rpmES)
        rpmGT=[]
        rpmES=[]
        plt.title(method)
        a+=1
    plt.show()

def comparison_Vgraph(database = "cohface", type="MAE", value="Farneback"):
    for metric in value:
        a=1
        plt.figure(figsize=(12, 6))
        for method in type:
            plt.subplot(1,len(type),a)
            df = pd.read_hdf("Results\\"+database+"_res_lin_"+method+".h5", key="df")
            df.dropna(inplace=True)
            z=df[metric]
            list1=[]
            for num in z:
                list1.append(float(num))
            
            sns.violinplot(y=list1, orient='v')
            plt.xlabel(method)
            if(a==1): plt.ylabel(metric)
            if(database=="cohface"):
                if((metric=="MAE") or (metric=="RMSE")): plt.ylim([-2,11])
                elif(metric=="processTime"): plt.ylim([0.2,9])
                elif(metric=="PCC"):plt.ylim([-1.5,1.5])
                elif(metric=="CCC"): plt.ylim([-.5,1.35])
            else:
                if((metric=="MAE") or (metric=="RMSE")): plt.ylim([-6,18])
                elif(metric=="processTime"): plt.ylim([0.2,5])
                elif(metric=="PCC"):plt.ylim([-1.5,1.5])
                elif(metric=="CCC"): plt.ylim([-.5,1.35])

            a+=1   
        plt.show()
def comparison_Vgraph_sig(database, type, value):
    for metric in value:
        a=1
        plt.figure(figsize=(12, 6))
        for method in type:
            plt.subplot(1,len(type),a)
            df = pd.read_hdf("Results_sig\\"+database+"_res_lin_"+method+".h5", key="df")
            df.dropna(inplace=True)
            z=df[metric]
            list1=[]
            for num in z:
                list1.append(float(num))
            
            sns.violinplot(y=list1, orient='v')
            plt.xlabel(method)
            if(a==1): plt.ylabel(metric)
            if(database=="cohface"):
                if((metric=="MAE") or (metric=="RMSE")): plt.ylim([-2,11])
                elif(metric=="processTime"): plt.ylim([0.2,9])
                elif(metric=="PCC"):plt.ylim([-1.5,1.5])
                elif(metric=="CCC"): plt.ylim([-.5,1.35])
            else:
                if(metric=="MAE"): plt.ylim([-7,25])
                if(metric=="RMSE"): plt.ylim([-7,25])
                elif(metric=="processTime"): plt.ylim([0.2,5])
                elif(metric=="PCC"):plt.ylim([-1.5,1.5])
                elif(metric=="CCC"): plt.ylim([-.5,1.35])

            a+=1   
        plt.show()

def comparison_Regplot(database = "cohface", type="MAE", value="Farneback"):
    plt.figure(figsize=(16, 10))
    a=1
    for method in type:
        plt.subplot(3,3,a)
        df = pd.read_hdf("Results\\"+database+"_res_lin_"+method+".h5", key="df")
        list1 = get_meanData(df,"rpmGT")
        list2 = get_meanData(df,"rpmES")
        sns.regplot(x=list1, y=list2)    
        x = np.linspace(10, 25)
        plt.plot(x,x,'r')
        plt.grid()
        plt.title(method + " , corr:" + str(round(ss.pearsonr(list1,list2).statistic,2)))
        a+=1
    plt.show()

def comparison_Regplot_sig(database = "cohface", type="MAE", value="Farneback"):
    plt.figure(figsize=(16, 10))
    a=1
    for method in type:
        plt.subplot(3,3,a)
        df = pd.read_hdf("Results_sig\\"+database+"_res_lin_"+method+".h5", key="df")
        list1 = get_meanData(df,"rpmGT")
        list2 = get_meanData(df,"rpmES")
        sns.regplot(x=list1, y=list2)    
        x = np.linspace(10, 25)
        plt.plot(x,x,'r')
        plt.grid()
        plt.title(method + " , corr:" + str(round(ss.pearsonr(list1,list2).statistic,2)))
        a+=1
    plt.show()

def comparison_AutoRankplot(database, type, value):
    plt.figure(figsize=(16, 10))
    list=[]
    for metric in value:
        data = pd.DataFrame()
        for method in type:
            df = pd.read_hdf("Results_sig\\" +database+ "_res_lin_"+method+".h5", key="df")
            for num in df[metric]:
                if(float(num)!=np.nan): list.append(float(num))
            data[method] = list
            list=[]
        if type == "MAE" or type == "RMSE" or type == "processtime":
            result = autorank(data, alpha=0.05, verbose=False, order= 'ascending')
        else:
            result = autorank(data, alpha=0.05, verbose=False, order= 'descending')
        #print(result.rankdf.meanrank)
        #results.append([result.rankdf.meanrank])
        plot_stats(result)
        plt.title(metric)
        plt.show()

def comparison_AutoRankplot_sig(database, type, value, order):
    plt.figure(figsize=(16, 10))
    list=[]
    for metric in value:
        data = pd.DataFrame()
        for method in type:
            df = pd.read_hdf("Results_sig\\" +database+ "_res_lin_"+method+".h5", key="df")
            for num in df[metric]:
                if(float(num)!=np.nan) or (float(num)<1000): list.append(float(num))
            data[method] = list
            list=[]
        result = autorank(data, alpha=0.05, verbose=False, order= order)
        #if type == "MAE" or type == "RMSE" or type == "processtime":
        #    result = autorank(data, alpha=0.05, verbose=False, order= 'descending')
        #else:
        #    result = autorank(data, alpha=0.05, verbose=False, order= 'ascending')
        #create_report(result)
        #print(result.rankdf.meanrank)
        #results.append([result.rankdf.meanrank])
        plot_stats(result)
        plt.title(metric)
        plt.show()
        print(result)
        #create_report(res)
        #latex_table(result)
