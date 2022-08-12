#!/usr/bin/env python
# coding: utf-8

import os
import alignWavFiles
from configobj import ConfigObj
from scipy.io import wavfile
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import wave
import IPython.display as ipd
from multiprocessing import Pool
from multiprocessing import cpu_count
import matplotlib.mlab as mlab
import AuditorySpectrogram as audspec


def getSpectralData(filename,cfgObj):
    fs,data = wavfile.read(filename)
    if fs != int(cfgObj.get('sampling_freq')):
        logging.error('Sampling frequency of file %s does not match sampling_freq in config file' % filename)
        sys.exit(1)
    spec = computeMelSpectrogram(data,cfgObj) # mel spectrogram

    return (spec,0,0)        
    #return (spec,minBinCenterFreq,binWidth)
    
def computeSpectrogram(d,cfgObj):
    '''
    Computes spectrogram on the given array of time-series values.  The necessary parameters
    for the computation are specified in the given ConfigParser object
    '''
    fs = int(cfgObj.get('sampling_freq')) # in Hz
    hop_size = float(cfgObj.get('hop_size')) # in sec
    fft_size = int(cfgObj.get('fft_size'))
    win_length = float(cfgObj.get('win_size')) # in sec
    win_samples = int(round(win_length*fs))
    noverlap_samples = win_samples - int(round(hop_size*fs))
    (Pxx,f,t) = mlab.specgram(d,NFFT=win_samples,Fs=fs,noverlap=noverlap_samples,pad_to=fft_size)
    return Pxx

def computeMelSpectrogram(d,cfgObj):
    '''
    Computes mel spectrogram on the given array of time-series values.  The necessary
    parameters for the computation are specified in the given ConfigParser object
    '''
    fs = int(cfgObj.get('sampling_freq')) # in Hz
    nfilts = int(cfgObj.get('num_bands'))
    minfreq = int(cfgObj.get('min_freq'))
    maxfreq = int(cfgObj.get('max_freq'))
    pspec = computeSpectrogram(d,cfgObj) # linear spectrogram
    (aspec,wts) = audspec.audspec(pspec,fs,nfilts,'mel',minfreq,maxfreq,1,1.0)
    return aspec

def getRandomSampleCov(audiofile,segment_len,cfgObj):
    # compute mel spectrogram of random segment
    (rate,d) = wavfile.read(audiofile)
    num_context_frames = int(cfgObj.get('num_context_frames'))
    total_nsamples = d.shape[0]
    if segment_len < 0:
        segment_nsamples = total_nsamples
    else:
        segment_nsamples = int(np.floor(segment_len * rate))
    # to avoid memory issues with long files, we cap the random segment to max_audio_blocksize
    max_audio_blocksize_frames = int(cfgObj.get('max_audio_blocksize'))
    hop_size = float(cfgObj.get('hop_size'))
    cap_nsamples = int(np.round(max_audio_blocksize_frames * hop_size * rate))
    start_idx = np.random.randint(max(total_nsamples-segment_nsamples+1,1))
    end_idx = min(start_idx + segment_nsamples,total_nsamples)
    melspec = computeMelSpectrogram(d[start_idx:end_idx],cfgObj)
    melspec[np.where(melspec==0)] = np.max(melspec)*1e-6 # handle 0 elements for logarithm
    logmelspec = np.log10(melspec)

    # compute covariance
    nSpecBands = logmelspec.shape[0]
    nSpecFrms = logmelspec.shape[1]
    mat = np.zeros((nSpecBands*num_context_frames,nSpecFrms-num_context_frames+1))
    for i in range(num_context_frames):
        mat[i*nSpecBands:(i+1)*nSpecBands,:] = logmelspec[:,i:i+nSpecFrms-num_context_frames+1]
    covEst = np.cov(mat)

    return (covEst,mat.shape[1])

def determineOptHalfFilters(filelist,cfgObj, return_eigvals=False):
    # first get covariance estimate based on random samples from files
    num_context_frames = int(cfgObj.get('num_context_frames'))
    num_freq_bands = int(cfgObj.get('num_bands'))
    len_sample = float(cfgObj.get('covar_estimate_samplelength'))
    n = num_context_frames * num_freq_bands
    accum = np.zeros((n,n))
    frameCnt = 0
    np.random.seed(0)
    for file in filelist:
        audiofile = file.strip()
        (covEst, nfrms) = getRandomSampleCov(audiofile,len_sample,cfgObj)
        accum += covEst * (nfrms-1)
        frameCnt += nfrms
    covarMatrix = accum / (frameCnt-1)
    
    # compute projected eigenvectors
    eigvals, eigvecs_unsorted = np.linalg.eig(covarMatrix)
    sortIdxs = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs_unsorted[:,sortIdxs] # eigvecs sorted by decreasing eigenvalues
    num_fp_bits = int(cfgObj.get('num_fingerprint_bits'))
    X = eigvecs[:,0:num_fp_bits]
    if return_eigvals:
        return np.sort(eigvals)[::-1]
    return X


# X specifies the mask matrix.  Each row specifies the weights of the spectrotemporal filter, and each
# column corresponds to a different filter.  There are num_bands * num_context_frames rows and num_fp_bits columns.
def computeProjectedEigenvectorFingerprints(filename,cfgObj,X):
    num_context = int(cfgObj.get('num_context_frames'))
    mask_filter_sep = float(cfgObj.get('mask_filter_sep'))
    hop_size = float(cfgObj.get('hop_size'))
    sep_nframes = np.round(mask_filter_sep / hop_size)
    num_fp_bits = X.shape[1]

    # compute mel spectrogram
    (spec,_,_) = getSpectralData(filename,cfgObj)
    spec = np.log10(spec)
    num_bands = spec.shape[0]
    num_frames = spec.shape[1]
    
    # create data matrix of each frame's context data -- each column corresponds to a context position, each row corresponds to a frame
    A = np.zeros((num_frames-num_context+1,num_context*num_bands))
    for i in range(num_context):
        A[:,i*num_bands:(i+1)*num_bands] = spec[:,i:i+num_frames-num_context+1].T
    # compute spectotemporal features using the (half) mask filters.  Each column corresponds to a feature, and each
    # row corresponds to a single frame.
    B = np.dot(A,X)
    # subtract a time-delayed version of each feature 
    C = B[0:int(B.shape[0]-sep_nframes),:] - B[int(sep_nframes):int(B.shape[0]),:]
    # threshold at 0
    D = C > 0
    # compute fingerprint value
    E = np.dot(D,np.power(2,np.arange(num_fp_bits))[::-1])


    # construct result
    tuples = []
    for i in range(E.shape[0]):
        tuples.append((int(E[i]),i))
    return tuples, C


def get_filter(ref_wave, cfgObj):
    return determineOptHalfFilters([ref_wave], cfgObj)


def get_hps_and_deltas(query_wave, cfgObj, maskMatrix=None, ref_wave=None):
    if maskMatrix is None:
        if ref_wave:
            maskMatrix = get_filter(ref_wave, cfgObj)
        else:
            return 'Need reference filepath when no filter is passed in.'
    hashprints, deltas = computeProjectedEigenvectorFingerprints(query_wave, cfgObj, maskMatrix)
    hashprints = list(list(zip(*hashprints))[0])
    return hashprints, deltas.T


def calculate_tamper_score(query_hps, ref_hps):
    offset = find_offset(query_hps, ref_hps)
    score = calculate_score(query_hps, ref_hps, offset) # substitute with proper scoring function
    return score, offset


def get_ref_chunk(C_ref, offset, query_hps):
    return C_ref[:,offset:offset+len(query_hps)]

