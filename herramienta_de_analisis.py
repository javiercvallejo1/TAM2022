import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.filter_design import butter
from scipy.signal import butter, lfilter, resample, lfilter
from scipy.fft import rfft, rfftfreq
import scipy.io


# 1 RMS
def RMS_ventaneado(window, signal, freq_sample):
    T = np.arange(window/freq_sample, len(signal) /
                  freq_sample, window/freq_sample)
    RMS = []
    f = 0
    i = 0
    for i in range(int(len(signal)/window)):
        s = 0
        for f in range(window):
            s = s + signal[f+(i*window)]**2
        l = np.sqrt(s/window)
        RMS.append(l)
    return RMS


# 2 Desv. estandar
def std_ventaneado(window, signal, freq_sample):
    T = np.arange(window/freq_sample, len(signal) /
                  freq_sample, window/freq_sample)
    std = []
    f = 0
    i = 0
    c = 0
    for i in range(int(len(signal)/window)):
        s = 0
        lista = []
        for f in range(window):
            lista.append(signal[f+(i*window)])
        mean = np.mean(lista)
        for c in range(window):
            s = s + (signal[c+(i*window)]-mean)**2
        l = np.sqrt((s/(window-1)))
        std.append(l)
    return std

# 3 Media


def mean_ventaneado(window, signal, freq_sample):
    T = np.arange(window/freq_sample, len(signal) /
                  freq_sample, window/freq_sample)
    mean = []
    f = 0
    i = 0
    for i in range(int(len(signal)/window)):
        s = 0
        for f in range(window):
            s = s + signal[f+(i*window)]
        l = (s/window)
        mean.append(l)
    return mean

# 4 pico pico


def PK_ventaneado(window, signal, freq_sample):
    T = np.arange(window/freq_sample, len(signal) /
                  freq_sample, window/freq_sample)
    PK = []
    f = 0
    i = 0
    for i in range(int(len(signal)/window)):
        s = 0
        lt = []
        for f in range(window):
            lt.append(signal[f+(i*window)])
        d = np.min(lt)
        f = np.max(lt)
        l = f - d
        PK.append(l)
    return PK

# 5 Kourtosis


def KTS_ventaneado(window, signal, freq_sample):
    T = np.arange(window/freq_sample, len(signal) /
                  freq_sample, window/freq_sample)
    KTS = []
    f = 0
    i = 0
    c = 0
    for i in range(int(len(signal)/window)):
        s = 0
        lista = []
        for f in range(window):
            lista.append(signal[f+(i*window)])
        mean = np.mean(lista)
        desv = np.std(lista)
        for c in range(window):
            s = s + ((signal[c+(i*window)]-mean)/desv)**4
        l = s*((window*(window+1))/((window-1)*(window-2)*(window-3))) - \
            ((3*(window-1)**2)/((window-2)*(window-3)))
        KTS.append(l)
    return KTS

# 6 asimetria


def SKS_ventaneado(window, signal, freq_sample):
    T = np.arange(window/freq_sample, len(signal) /
                  freq_sample, window/freq_sample)
    SKS = []
    f = 0
    i = 0
    c = 0
    for i in range(int(len(signal)/window)):
        s = 0
        lista = []
    for f in range(window):
        lista.append(signal[f+(i*window)])
        mean = np.mean(lista)
        desv = np.std(lista)
    for c in range(window):
        s = s + ((signal[c+(i*window)]-mean)/desv)**3
        l = s*(window/(window-1)*(window-2))
        SKS.append(l)
    return SKS

# 7 fourier transform


def Fourier_transform(signal, freq_sample):
    N = signal.size
    yf = rfft(signal)
    xf = rfftfreq(N, 1/freq_sample)

    plt.figure(figsize=(10, 5))
    plt.title("Transformada de Fourier de la señal",
              fontdict={'family': 'monospace', 'weight': 'bold', 'size': 10})
    plt.grid()
    plt.plot(xf, np.abs(yf))
    plt.show()


filepath = r'data\Datos_caso_A.csv'


f = amp.size
sampling_rate = 20000  # sampling rate, since we dont have a time reference, is mandatory
sampling_period = 1/sampling_rate  # time spacing between two samples
max_time = f/sampling_rate
time = np.arange(0, max_time, sampling_period)


df = pd.read_csv(filepath)  # CHANGE THE FILEPATH

print(df.head())
