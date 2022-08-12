#!/usr/local/bin/python

'''
Functions to compute an auditory spectrogram.  Translated from Dan Ellis implementation
in Matlab.
'''

import numpy as np
import logging
import sys

def audspec(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth):
    '''
    Perform critical band analysis
    Returns (aspectrum,wts)
    '''
    (nfreqs,nframes) = pspectrum.shape
    nfft = (nfreqs-1)*2

    if fbtype == 'mel':
        (wts,_) = fft2melmx(nfft,sr,nfilts,bwidth,minfreq,maxfreq)
    else:
        logging.error('fbtype %s not recognized' % fbtype)
        sys.exit(1)

    wts = wts[:,0:nfreqs]

    if sumpower:
        aspectrum = np.dot(wts,pspectrum)
    else:
        aspectrum = np.dot(wts,np.sqrt(pspectrum))**2

    return (aspectrum,wts)

def fft2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel=0, constamp=0):
    '''
    Documentation copied from Dan Ellis matlab implementation
    
    function [wts,binfrqs] = fft2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel, constamp)
    % [wts,frqs] = fft2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel, constamp)
    %      Generate a matrix of weights to combine FFT bins into Mel
    %      bins.  nfft defines the source FFT size at sampling rate sr.
    %      Optional nfilts specifies the number of output bands required
    %      (else one per "mel/width"), and width is the constant width of each
    %      band relative to standard Mel (default 1).
    %      While wts has nfft columns, the second half are all zero.
    %      Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
    %      minfrq is the frequency (in Hz) of the lowest band edge;
    %      default is 0, but 133.33 is a common standard (to skip LF).
    %      maxfrq is frequency in Hz of upper edge; default sr/2.
    %      You can exactly duplicate the mel matrix in Slaney`s mfcc.m
    %      as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    %      htkmel=1 means use HTK`s version of the mel curve, not Slaney`s.
    %      constamp=1 means make integration windows peak at 1, not sum to 1.
    %      frqs returns bin center frqs.
    % 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx
    '''
    if nfilts == 0:
        nfilts = int(np.ceil(hz2mel(maxfrq,htkmel)/2))

    wts = np.zeros((nfilts,nfft))

    # center freqs of each fft bin
    fftfrqs = np.arange(0,nfft/2+1,1)*1.0/nfft*sr

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfrq, htkmel)
    maxmel = hz2mel(maxfrq, htkmel)
    binfrqs = mel2hz(minmel+np.arange(0,nfilts+2,1)*1.0/(nfilts+1)*(maxmel-minmel), htkmel)

    binbin = np.round(binfrqs*1.0/sr*(nfft-1))

    for i in range(nfilts):
        fs = binfrqs[i+np.array([0,1,2])]
        # scale by width
        fs = fs[1]+width*(fs - fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - fs[0])/(fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs)/(fs[2] - fs[1])
        # .. then intersect them with each other and zero
        wts[i,0:int(nfft/2+1)] = np.maximum(0,np.minimum(loslope, hislope))
        
        # actual algo and weighting in feacalc (more or less)
        #  wts(i,:) = 0;
        #  ww = binbin(i+2)-binbin(i);
        #  usl = binbin(i+1)-binbin(i);
        #  wts(i,1+binbin(i)+[1:usl]) = 2/ww * [1:usl]/usl;
        #  dsl = binbin(i+2)-binbin(i+1);
        #  wts(i,1+binbin(i+1)+[1:(dsl-1)]) = 2/ww * [(dsl-1):-1:1]/dsl;
        # need to disable weighting below if you use this one

    if constamp == 0:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = np.dot(np.diag(2/(binfrqs[np.arange(2,nfilts+2)]-binfrqs[np.arange(0,nfilts)])),wts)

    # Make sure 2nd half of FFT is zero
    wts[:,int(nfft/2+1):] = 0
    # seems like a good idea to avoid aliasing

    return (wts,binbin)
#    return (wts,binfrqs)

def mel2hz(z,htk=0):
    '''
    function f = mel2hz(z, htk)
    %   f = mel2hz(z, htk)
    %   Convert `mel scale` frequencies into Hz
    %   Optional htk = 1 means use the HTK formula
    %   else use the formula from Slaney`s mfcc.m
    % 2005-04-19 dpwe@ee.columbia.edu
    '''
    if htk==1:
        f = 700.0*(np.power(10.0,z/2595.0)-1)
    else:

        f_0 = 0.0 # 133.33333;
        f_sp = 200.0/3 # 66.66667;
        brkfrq = 1000.0
        brkpt  = (brkfrq - f_0)/f_sp # starting mel value for log region
        logstep = np.exp(np.log(6.4)/27.0); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        if hasattr(z,'__getitem__'):
            linpts = np.nonzero(z < brkpt)
            logpts = np.nonzero(np.logical_not(z < brkpt))
        
            f = 0*z;
    
            # fill in parts separately
            f[linpts] = f_0 + f_sp*z[linpts]
            f[logpts] = brkfrq*np.exp(np.log(logstep)*(z[logpts]-brkpt))
        else:
            if z < brkpt:
                f = f_0 + f_sp*z
            else:
                brkfrq*np.exp(np.log(logstep)*(z-brkpt))
        
    return f

def hz2mel(f,htk=0):
    '''
    function z = hz2mel(f,htk)
    %  z = hz2mel(f,htk)
    %  Convert frequencies f (in Hz) to mel 'scale'.
    %  Optional htk = 1 uses the mel axis defined in the HTKBook
    %  otherwise use Slaney`s formula
    % 2005-04-19 dpwe@ee.columbia.edu
    '''
    if htk==1:
        z = 2595.0 * np.log10(1.0+f/700.0)
    else:
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m

        f_0 = 0.0 # 133.33333;
        f_sp = 200.0/3 # 66.66667;
        brkfrq = 1000.0
        brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
        logstep = np.exp(np.log(6.4)/27.0) # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        if hasattr(f,'__getitem__'):

            linpts = np.nonzero(f < brkfrq)
            logpts = np.nonzero(np.logical_not(f < brkfrq))

            z = 0*f

            # fill in parts separately
            z[linpts] = (f[linpts] - f_0)/f_sp
            z[logpts] = brkpt+(np.log(f[logpts]/brkfrq))/np.log(logstep)
        else:
            if f < brkfrq:
                z = (f-f_0)/f_sp
            else:
                z = brkpt+(np.log(f/brkfrq))/np.log(logstep)

    return z

def audspec_mdct(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth):
    '''
    Perform critical band analysis on MDCT coefficients
    Returns (aspectrum,wts)
    '''
    (nfreqs,nframes) = pspectrum.shape
    #nfft = (nfreqs-1)*2

    if fbtype == 'mel':
        #(wts,_) = mdct2melmx(nfft,sr,nfilts,bwidth,minfreq,maxfreq)
        (wts,_) = mdct2melmx(nfreqs*2,sr,nfilts,bwidth,minfreq,maxfreq)
    else:
        logging.error('fbtype %s not recognized' % fbtype)
        sys.exit(1)

    wts = wts[:,0:nfreqs]

    if sumpower:
        aspectrum = np.dot(wts,pspectrum)
    else:
        aspectrum = np.dot(wts,np.sqrt(pspectrum))**2

    return (aspectrum,wts)

def mdct2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel=0, constamp=0):
    '''
    Copied from fft2melmx and modified appropriately
    '''
    if nfilts == 0:
        nfilts = int(np.ceil(hz2mel(maxfrq,htkmel)/2))

    wts = np.zeros((nfilts,nfft))

    # center freqs of each fft bin
    binFreqWidth = 1.0*sr/nfft
    fftfrqs = (np.arange(0,nfft/2,1)+0.5)*1.0*binFreqWidth

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfrq, htkmel)
    maxmel = hz2mel(maxfrq, htkmel)
    binfrqs = mel2hz(minmel+np.arange(0,nfilts+2,1)*1.0/(nfilts+1)*(maxmel-minmel), htkmel)

    binbin = np.round(binfrqs*1.0/sr*(nfft-1)-.5)

    for i in range(nfilts):
        fs = binfrqs[i+np.array([0,1,2])]
        # scale by width
        fs = fs[1]+width*(fs - fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - fs[0])/(fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs)/(fs[2] - fs[1])
        # .. then intersect them with each other and zero
        wts[i,0:nfft/2] = np.maximum(0,np.minimum(loslope, hislope))
        
        # actual algo and weighting in feacalc (more or less)
        #  wts(i,:) = 0;
        #  ww = binbin(i+2)-binbin(i);
        #  usl = binbin(i+1)-binbin(i);
        #  wts(i,1+binbin(i)+[1:usl]) = 2/ww * [1:usl]/usl;
        #  dsl = binbin(i+2)-binbin(i+1);
        #  wts(i,1+binbin(i+1)+[1:(dsl-1)]) = 2/ww * [(dsl-1):-1:1]/dsl;
        # need to disable weighting below if you use this one

    if constamp == 0:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = np.dot(np.diag(2/(binfrqs[np.arange(2,nfilts+2)]-binfrqs[np.arange(0,nfilts)])),wts)

    # Make sure 2nd half of FFT is zero
    wts[:,(nfft/2):] = 0
    # seems like a good idea to avoid aliasing

    return (wts,binbin)
#    return (wts,binfrqs)
