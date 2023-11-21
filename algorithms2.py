# import h5py
import cv2 as cv
import numpy as np
import pandas as pd

# import mediapipe as mp
from scipy import signal
from PIL import Image, ImageStat
from scipy.signal._arraytools import even_ext
from importlib import import_module, util

from inspect import getmembers, isfunction
import os


import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets


from utils import *
from videoUtil import *
from opticalFlow import *


def lin(video_path, dataset, of_type):
    """
    Questo metodo applica Lin et al. algoritmo per la misurazione del respiro

     Parametri
     ----------
         video_path : percorso del video da cui viene estratto il segnale respiratorio

     ritorna
     -------
         il segnale respiratorio stimato
    """
    # implementati = 'Farneback', 'Lucaskanade_dense', "RLOF", "tvl1"
    # da implementare = "deepflow", "Simple", 'mrflow', "epicflow", 'HornSchunck' , "Brox"
    available_datasets = ["cohface", "bp4d"]
    available_op_type = [
        "RLOF",
        "Lucaskanade_dense",
        "tvl1",
        "Farneback",
        "deepflow",
        "mrflow",
        "epicflow",
        "HornSchunck",
        "simple",
        "Dense_Inverse_Search",
        "Brox",
        "Dense_Inverse_Search_UFast",
        "PCA",
    ]

    assert of_type in available_op_type, "\nrPPG optical flow not recognized!!"
    assert dataset in available_datasets, "\nrPPG method not recognized!!"

    if dataset == "bp4d":
        fps = 25
    else:
        fps = get_fps(video_path)

    i = 0
    prev = 0
    curr = 0
    median = []
    mp_pose = mp.solutions.pose

    # Run MediaPipe Pose and draw pose landmarks.
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2
    ) as pose:

        for frame in extract_frames_yield(video_path):
            # print('\rframe number ',i, end='', flush=True)#debug
            total = get_Totfps(video_path) - 1
            print("\r%d%% complete" % int(i / total * 100), end="", flush=True)  # dubug
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            image_height, image_width, _ = frame.shape

            # Get landmark.
            if results.pose_landmarks is None:
                x_left = 0
                y_left = 0
                x_right = 0
                y_right = 0
                print("None landmark")

            else:
                # Get landmark.
                x_left = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].x
                    * image_width
                )
                y_left = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].y
                    * image_height
                )
                x_right = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].x
                    * image_width
                )
                y_right = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].y
                    * image_height
                )

            if i == 0:
                patch_width = x_left - x_right
                patch_height = image_height - (round(y_left) - 70)

            im = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            im = Image.fromarray(frame)

            # Crop chest ROIs
            if dataset == "bp4d":
                x_right = 0
                x_left = image_width - 1
                y_right = image_height - 200
                y_left = image_height - 1
                chest = im.crop(box=(x_right, y_right, x_left, y_left))
            else:
                chest = im.crop(
                    box=(
                        round(x_right) + patch_width / 6,
                        image_height - patch_height,
                        round(x_right) + 5 / 6 * patch_width,
                        image_height,
                    )
                )

            if i == 0:
                prev = np.asarray(cv.cvtColor(np.asarray(chest), cv.COLOR_BGR2GRAY))
                i += 1
                continue

            # cv.imshow('Frame', frame)

            # get opt flow
            gray = cv.cvtColor(np.asarray(chest), cv.COLOR_BGR2GRAY)
            curr = gray
            # I.append(im) #debug
            # C.append(gray) #debug
            flow = np.zeros_like(prev)

            # scelgo l'optical flow da utilizzare
            # Calculates dense optical flow by Farneback method
            if of_type == "Farneback":
                # Calculates dense optical flow by Farneback method
                flow = cv.calcOpticalFlowFarneback(
                    prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            elif of_type == "HornSchunck":
                # Calculates dense optical flow by HornSchunck method
                U,V = HS(prev, curr, .8, 10)
                

            elif of_type == "Lucaskanade_dense":
                flow = cv.optflow.calcOpticalFlowSparseToDense(
                    curr, prev, grid_step=5, sigma=0.5
                )

            elif of_type == "RLOF":
                # Calcolo rapido del flusso ottico denso basato su robusti algoritmi di flusso ottico locale (RLOF) e schema di interpolazione da sparso a denso.
                # Crea un oggetto RLOFOpticalFlowParameter con i valori desiderati
                rlofParam = cv.optflow.RLOFOpticalFlowParameter_create()
                rlofParam.setSupportRegionType(cv.optflow.SR_FIXED)
                # Chiama la funzione calcOpticalFlowDenseRLOF
                flow = cv.optflow.calcOpticalFlowDenseRLOF(
                    curr, prev, flow, rlofParam, forwardBackwardThreshold=1
                )
                # flow = cv.optflow.calcOpticalFlowDenseRLOF(prev, curr, flow, interp_type=cv.optflow.INTERP_GEO)
            
            elif of_type == "tvl1":
                dtvl1 = cv.optflow.DualTVL1OpticalFlow_create()
                dtvl1.setLambda(0.15)
                dtvl1.setTheta(0.3)
                dtvl1.setScalesNumber(5)
                dtvl1.setWarpingsNumber(5)
                # Calcola il campo di flusso ottico
                flow = dtvl1.calc(prev, curr, None)

            elif of_type == "deepflow":
                deepflow = cv.optflow.createOptFlow_DeepFlow()
                flow = deepflow.calc(prev, curr, None)

            elif of_type == "simple":
                #sFlow=cv.optflow.createOptFlow_SimpleFlow()
                #prev = cv.cvtColor(prev, cv.COLOR_GRAY2RGB);
                #curr = cv.cvtColor(curr, cv.COLOR_GRAY2RGB);	
                # flow	=	cv.optflow.calcOpticalFlowSF(from, to, layers, averaging_block_size, max_flow) 
                # Parameters
                    # from	First 8-bit 3-channel image.
                    # to	Second 8-bit 3-channel image of the same size as prev
                    # flow	computed flow image that has the same size as prev and type CV_32FC2
                    # layers	Number of layers
                    # averaging_block_size	Size of block through which we sum up when calculate cost function for pixel
                    # max_flow	maximal flow that we search at each level
                    # sigma_dist	vector smooth spatial sigma parameter
                    # sigma_color	vector smooth color sigma parameter
                    # postprocess_window	window size for postprocess cross bilateral filter
                    # sigma_dist_fix	spatial sigma for postprocess cross bilateralf filter
                    # sigma_color_fix	color sigma for postprocess cross bilateral filter
                    # occ_thr	threshold for detecting occlusions
                    # upscale_averaging_radius	window size for bilateral upscale operation
                    # upscale_sigma_dist	spatial sigma for bilateral upscale operation
                    # upscale_sigma_color	color sigma for bilateral upscale operation
                    # speed_up_thr	threshold to detect point with irregular flow - where flow should be recalculated after upscale
                #calcOpticalFlowSF(prev, curr,flow,3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
                #flow = cv.optflow.calcOpticalFlowSF(prev, curr, 3, 2, 4)
                
                cv.optflow.calcOpticalFlowSF(prev, curr,flow,3, 2, 4)
                

            elif of_type == "Dense_Inverse_Search":
                # PRESET_ULTRAFAST = 0,
                # PRESET_FAST = 1,
                # PRESET_MEDIUM = 2
                # Questa classe implementa l'algoritmo di flusso ottico DIS (Dense Inverse Search).
                # Maggiori dettagli sull'algoritmo sono disponibili all'indirizzo https://docs.opencv.org/3.4/d0/de3/citelist.html#CITEREF_Kroeger2016 .
                # Include tre preset con parametri preselezionati per fornire un ragionevole compromesso tra velocità e qualità.
                disflow = cv.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = disflow.calc(prev, curr, None)

            elif of_type == "Dense_Inverse_Search_UFast":
                disflow = cv.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                flow = disflow.calc(prev, curr, None)

            elif of_type == "PCA":
                PCAflow = cv.optflow.createOptFlow_PCAFlow()
                flow = PCAflow.calc(prev, curr, None)

            
            if of_type == "HornSchunck":
                vert= V
            else:
                vert = flow[..., 1]
            # vert_img = cv.normalize(flow[...,1], None, 0, 255, cv.NORM_MINMAX)
            # vert_img = vert.astype('uint8')

            vert = vert.flatten()
            #vert = np.sort(vert)
            median.append(np.median(vert))

            prev = curr
            i += 1
        # showVideo(I)#debug
        # showVideo(C)#debug
    #salvo median su db e poi faccio i calcoli average_filter(median)
    return median


def showVideo(I):
    n = len(I)

    def view_image(idx):
        plt.imshow(I[idx - 1], interpolation="nearest", cmap="gray")

    interact(view_image, idx=widgets.IntSlider(min=1, max=n, step=1, value=1))


def lin_EVM(video_path, dataset):
    """
    This method applies Lin et al. algorithm for breath measurement

    Parameters
    ----------
        path : path of the video from which the respiratory signal is extracted

    Returns
    -------
        the estimated respiratory signal
    """

    available_datasets = ["cohface_EVM_mov"]
    assert dataset in available_datasets, "\nrPPG method not recognized!!"

    fps = get_fps(video_path)
    prev = 0
    curr = 0
    patch_width = 300
    xy_coordinates = []
    median = []
    mp_pose = mp.solutions.pose
    # ins = pixellib.torchbackend.instance()
    ins = instanceSegmentation()
    ins.load_model("/home/studenti/lombarda/pointrend_resnet50.pkl")

    # Run MediaPipe Pose and draw pose landmarks.
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2
    ) as pose:
        for frame in extract_frames_yield(video_path):
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # Get landmark.
            if results.pose_landmarks is None:
                xy_coordinates.append([0, 0, 0, 0])
            else:
                image_height, image_width, _ = frame.shape
                x_left = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].x
                    * image_width
                )
                y_left = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].y
                    * image_height
                )
                x_right = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].x
                    * image_width
                )
                y_right = (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].y
                    * image_height
                )
                xy_coordinates.append([x_left, y_left, x_right, y_right])

        # set as reference coordinates the ones of the first frame
        x_left_ref = xy_coordinates[0][0]
        y_left_ref = xy_coordinates[0][1]
        x_right_ref = xy_coordinates[0][2]
        y_right_ref = xy_coordinates[0][3]

        # drop coordinates bad identified by mediapipe
        for i in np.arange(len(xy_coordinates)):
            if (
                xy_coordinates[i][0] < x_left_ref - 80
                or xy_coordinates[i][0] > x_left_ref + 80
                or xy_coordinates[i][1] < y_left_ref - 80
                or xy_coordinates[i][1] > y_left_ref + 80
                or xy_coordinates[i][2] < x_right_ref - 80
                or xy_coordinates[i][2] > x_right_ref + 80
                or xy_coordinates[i][3] < y_right_ref - 80
                or xy_coordinates[i][3] > y_right_ref + 80
            ):
                xy_coordinates[i] = [0, 0, 0, 0]

        for i in np.arange(len(xy_coordinates)):
            prev = 1
            nxt = 1
            x_left, y_left, x_right, y_right = xy_coordinates[i]
            # interpolate to get meaningful coordinates
            while [x_left, y_left, x_right, y_right] == [0, 0, 0, 0]:
                if i + nxt < len(xy_coordinates) and xy_coordinates[i + nxt] == [
                    0,
                    0,
                    0,
                    0,
                ]:
                    nxt += 1
                    continue

                if i - prev >= 0 and xy_coordinates[i - prev] == [0, 0, 0, 0]:
                    prev += 1
                    continue

                if i + nxt >= len(xy_coordinates) or i - prev < 0:
                    x_left = xy_coordinates[i - 1][0]
                    y_left = xy_coordinates[i - 1][1]
                    x_right = xy_coordinates[i - 1][2]
                    y_right = xy_coordinates[i - 1][3]

                else:
                    x_left = (
                        xy_coordinates[i + nxt][0] + xy_coordinates[i - prev][0]
                    ) / 2
                    y_left = (
                        xy_coordinates[i + nxt][1] + xy_coordinates[i - prev][1]
                    ) / 2
                    x_right = (
                        xy_coordinates[i + nxt][2] + xy_coordinates[i - prev][2]
                    ) / 2
                    y_right = (
                        xy_coordinates[i + nxt][3] + xy_coordinates[i - prev][3]
                    ) / 2

            xy_coordinates[i] = [x_left, y_left, x_right, y_right]

        i = 0
        for frame in extract_frames_yield(video_path):
            if i == 0:
                patch_width = x_left - x_right
                patch_height = image_height - (round(y_left) - 70)

            im = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            # Crop chest ROIs
            chest = im.crop(
                box=(
                    round(x_right) + patch_width / 6,
                    image_height - patch_height,
                    round(x_right) + 5 / 6 * patch_width,
                    image_height,
                )
            )

            if i == 0:
                prev = np.asarray(cv.cvtColor(np.asarray(chest), cv.COLOR_BGR2GRAY))
                i += 1
                continue

            # get opt flow
            gray = cv.cvtColor(np.asarray(chest), cv.COLOR_BGR2GRAY)
            curr = gray

            # Calculates dense optical flow by Farneback method
            flow = cv.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            vert = flow[..., 1]
            vert_img = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
            vert_img = vert.astype("uint8")

            vert = vert.flatten()
            vert = np.sort(vert)
            median.append(np.median(vert))

            prev = curr
            i += 1
    return average_filter(median)


def get_shoulder_regions(mask):
    """
    This method returns dA and dB, expressing the number of pixels above and below the shoulder edge

    Parameters
    ----------
        mask : the mask highlighting the shoulder region

    Returns
    -------
        dA, dB
    """
    n_true = np.count_nonzero(mask)
    height, width = mask.shape
    n_total = height * width
    n_false = n_total - n_true

    return n_true, n_false


def get_dI(dA, dB):
    """
    This method computes the vertical movement of shoulder

    Parameters
    ----------
        dA : the amount of pixels above the shoulder edge
        dB : the amount of pixels below the shoulder edge

    Returns
    -------
        dI, sensitive to vertical movement
    """
    return (dA - dB) / (dA + dB)

    # / calculate dense flow
    if method == "Farneback":
        of_instance = cv.optflow.createOptFlow_Farneback()
    elif method == "DIS":
        of_instance = cv.optflow.createOptFlow_DIS()
    elif method == "DeepFlow":
        of_instance = cv.optflow.createOptFlow_DeepFlow()
    elif method == "PCAFlow":
        of_instance = cv.optflow.createOptFlow_PCAFlow()
    elif method == "Simple":
        of_instance = cv.optflow.createOptFlow_SimpleFlow()
    elif method == "SparseToDense":
        of_instance = cv.optflow.createOptFlow_SparseToDense()
