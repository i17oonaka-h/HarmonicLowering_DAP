import soundfile as sf
from scipy import signal
import pickle
import numpy as np
import scipy.fftpack as scft
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize

def _safe_db(num, den):
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)

def sisdr(ref, est):
    assert(est.shape[0]==ref.shape[0]), 'The shape of estimated sources and the true sources should match.'
    energy_s_true = np.sum(ref**2)
    lamb = np.sqrt(np.sum(ref**2)/np.sum(est**2))
    distortion = np.sum((ref-lamb*est)**2)
    sisdr_ret = _safe_db(energy_s_true, distortion)
    return sisdr_ret

def pad32(x):
    shape = x.shape
    pad1 = 32-shape[2]%32
    pad2 = 32-shape[3]%32
    m = nn.ZeroPad2d((pad2, 0, pad1, 0))
    return m(x), pad1, pad2

def wavread(fn):
    data, sr = sf.read(fn)
    f = sf.info(fn)
    return data, sr, f.subtype

def wavwrite(fn, data, sr, subtype, resr=None):
    if resr is None:
        sf.write(fn, data, sr, subtype)
    else:
        data = signal.resample(data, int(resr*len(data)/sr))
        sf.write(fn, data, resr, subtype)

def stft(x, win=np.hamming, fftl=512, shift=128, winl=None, onesided=True): #短時間フーリエ変換
    """
    x: target
    win: 窓関数の種類
    fftl: 
    """
    if winl is None:
        winl = fftl
    assert (winl <= fftl), "FFT length < window length."
    # winlで指定された長さの窓関数win
    win = np.pad(np.hamming(winl),[int(np.ceil((fftl-winl)/2)),int(np.floor((fftl-winl)/2))], 'constant')
    # new_x: 端点に余裕を持たせた(パティングした)xを生成
    l = len(x) 
    new_l = 2*(fftl-shift)+int(np.ceil(l/shift))*shift
    new_x = np.zeros(new_l)
    new_x[fftl-shift:fftl-shift+l] = x
    M = int((new_l-fftl+shift)/shift)
    X = np.zeros([M,fftl],dtype = np.complex128)
    shift_matrix = np.arange(0, shift*M, shift).reshape(M,1).repeat(fftl,1)
    index_matrix = np.arange(0, fftl)
    index_matrix = index_matrix + shift_matrix
    X = scft.fft(new_x[index_matrix]*win , fftl)
    if onesided: X = X[:,:fftl//2+1]
    return X

def istft(X, win=np.hamming, fftl=512, shift=128, winl=None, x_len=None, onesided=True):
    if winl is None:
        winl = fftl
    assert (winl <= fftl), "FFT length > window length."
    win = np.pad(np.hamming(winl),[int(np.ceil((fftl-winl)/2)),int(np.floor((fftl-winl)/2))], 'constant')
    if onesided:
        X = np.hstack((X, np.conjugate(np.fliplr(X[:,1:X.shape[1]-1]))))
    M, fftl = X.shape
    l = (M-1)*shift+fftl
    xx = scft.ifft(X).real * win
    xtmp = np.zeros(l, dtype = np.float64)
    wsum = np.zeros(l, dtype = np.float64)
    for m in range(M):
        start = shift*m                                                                     
        xtmp[start : start+fftl] += xx[m, :]
        wsum[start : start + fftl] += win ** 2
    pos = (wsum != 0)                                                  
    xtmp[pos] /= wsum[pos]
    x = xtmp[fftl-shift:-fftl+shift]
    if x_len is not None:
        x = x[:x_len]
    return x

def Const(s,n,snr):
    s_amp =np.abs(s)
    S = np.mean(s_amp**2)
    n_amp = np.abs(n)
    N = np.mean(n_amp**2)
    c = np.sqrt(S/(N*10**(snr/10)))
    return c

def create_mixture(clean, noise, snr, nonVoice_section=0):
    if len(clean)+nonVoice_section > len(noise):
        add_len = int((len(clean)+nonVoice_section) / len(noise)) +1
        noise_tmp = noise
        for i in range(add_len):
            noise = np.concatenate([noise,noise_tmp])
    
    noise = noise[0:len(clean)+nonVoice_section]
    noise = Const(clean,noise[nonVoice_section:],snr)*noise
    zero_list = [0] * nonVoice_section
    clean = np.insert(clean,0,zero_list)
    assert(len(clean) == len(noise)), f"The length clean({len(clean)}) and noise({len(noise)}) do not match."
    mixture_wave = clean + noise
    return mixture_wave, noise

def np_to_torch(ndarray_):
    return torch.from_numpy(ndarray_).clone()

def torch_to_np(tensor_):
    return tensor_.to('cpu').detach().numpy().copy()

def specshow(x, sr=16000, win=np.hamming, fftl=512, shift=128, winl=None, title='', mode='mesh', fig_size=(10,3), ax=None, c_map='jet', v_min=-70, v_max=40, save_path=None, colorbar=True):
    if winl is None:
        winl = fftl
    assert (winl <= fftl), "FFT length < window length."
    win = np.pad(np.hamming(winl),[int(np.ceil((fftl-winl)/2)),int(np.floor((fftl-winl)/2))], 'constant')
    if len(x.shape)==1:
        X = stft(x, win, fftl, shift)
        f = np.linspace(0, sr//2, X.shape[1])
        t = np.arange(X.shape[0])*shift/sr
    else:
        X = x[:,:fftl//2+1]
        f = np.linspace(0, sr//2, X.shape[1])
        t = np.arange(X.shape[0])*shift/sr
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        plt.tight_layout()
        if(mode=='mesh'):im=plt.pcolormesh(t, f, 20*np.log10(np.abs(X.T)+1e-8), cmap=c_map, norm=Normalize(vmin=v_min, vmax=v_max),zorder=-10)
        elif(mode=='imshow'):im=plt.imshow(20*np.log10(np.abs(X.T)+1e-8)[::-1,:], cmap=c_map, extent=[0,t[-1],0,f[-1]], aspect='auto', norm=Normalize(vmin=v_min, vmax=v_max),zorder=-10)
        plt.title(title)
        plt.ylabel('Freq. [Hz]')
        plt.xlabel('Time [sec]')
        if colorbar:
            plt.colorbar(format='%+2.0f dB')
        if(save_path is not None):
            matplotlib.interactive(False)
            plt.gca().set_rasterization_zorder(0)
            plt.tight_layout()
            plt.savefig('%s' %(save_path))
            #plt.show()
            plt.close(fig)
    else:
        if(mode=='mesh'):im=ax.pcolormesh(t, f, 20*np.log10(np.abs(X.T)+1e-8), cmap=c_map, norm=Normalize(vmin=v_min, vmax=v_max), zorder=-10)
        elif(mode=='imshow'):im=ax.imshow(20*np.log10(np.abs(X.T)+1e-8)[::-1,:], cmap=c_map, extent=[0,t[-1],0,f[-1]], aspect='auto', norm=Normalize(vmin=v_min, vmax=v_max), zorder=-10)
        ax.set_title(title)
        ax.set_ylabel('Freq. [Hz]')
        ax.set_xlabel('Time [sec]')
        if colorbar:
            plt.colorbar(im, format='%+2.0f dB', ax=ax)
        ax.set_rasterization_zorder(0)