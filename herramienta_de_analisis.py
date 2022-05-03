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




def calculate_time_vector(dataframe,sampling_rate):
    vector_size=len(dataframe)
    sampling_rate : int= sampling_rate  # sampling rate, since we dont have a time reference, is mandatory
    sampling_period = 1/sampling_rate  # time spacing between two samples
    max_time = vector_size/sampling_rate
    time = np.arange(0, max_time, sampling_period)
    return np.array(time)

def crest_factor(rms_vector,peak_vector):
    crest_vector=[]
    for rms, peak in zip(rms_vector,peak_vector):
    
        crest_factor= peak[0]/rms[0]
        crest_vector.append(crest_factor)
    return crest_vector


###################################################################################
filepath = r'data/Datos_caso_B.csv'
window = 20000
sampling_rate = 20000


df=pd.read_csv(filepath #COMMENT THIS ONE WHEN WORKING WITH MATLAB SOURCES

            #    ,names=[ "a(t)_0", 
            #             "a(t)_1",
            #             "a(t)_2",
            #             "a(t)_3",
            #             "v(t)_0",
            #             "v(t)_1"
            #             "v(t)_2",
            #             "v(t)_3",
            #             "x(t)_0",
            #             "x(t)_1",
            #             "x(t)_2",
            #             "x(t)_3"]
                        )


time= calculate_time_vector(df["a(t)_0"],20000)
df['time'] = time

acceleration = [ df["a(t)_0"]
                ,df["a(t)_1"]
                ,df["a(t)_2"]
                ,df["a(t)_3"]                                            
                ]

velocity=[df["v(t)_0"]
          ,df["v(t)_1"]
          ,df["v(t)_2"]
          ,df["v(t)_3"]]


displacement = [df["x(t)_0"]
                ,df["x(t)_1"]
                ,df["x(t)_2"]
                ,df["x(t)_3"]]



####  RMS
RMS_acceleration=[]

for i in acceleration:
    rms = RMS_ventaneado(window, i, sampling_rate)
    RMS_acceleration.append(rms)

print("RMS ACCELERATION",RMS_acceleration)



RMS_velocity = []

for i in velocity:
    rms = RMS_ventaneado(window, i, sampling_rate)
    RMS_velocity.append(rms)

print("RMS VELOCITY",RMS_velocity)

RMS_displacement = []

for i in acceleration:
    rms = RMS_ventaneado(window, i, sampling_rate)
    RMS_displacement.append(rms)

print("RMS DISPLACEMENT",RMS_displacement)

##### STD 

STD_acceleration=[]

for i in acceleration:
    std= std_ventaneado(window, i, sampling_rate)
    STD_acceleration.append(std)

print("STD ACCELERATION",STD_acceleration)


STD_velocity = []

for i in velocity:
    std = std_ventaneado(window, i, sampling_rate)
    STD_velocity.append(std)

print("STD VELOCITY",STD_velocity)

STD_displacement = []

for i in acceleration:
    std = std_ventaneado(window, i, sampling_rate)
    STD_displacement.append(std)

print("STD DISPLACEMENT",STD_displacement)

##### MEAN


mean_acceleration=[]

for i in acceleration:
    mean= mean_ventaneado(window, i, sampling_rate)
    mean_acceleration.append(mean)

print("mean ACCELERATION",mean_acceleration)


mean_velocity = []

for i in velocity:
    mean = mean_ventaneado(window, i, sampling_rate)
    mean_velocity.append(mean)

print("mean VELOCITY",mean_velocity)

mean_displacement = []

for i in acceleration:
    mean = mean_ventaneado(window, i, sampling_rate)
    mean_displacement.append(mean)

print("mean DISPLACEMENT",mean_displacement)


#####P2P

PK_acceleration=[]

for i in acceleration:
    PK= PK_ventaneado(window, i, sampling_rate)
    PK_acceleration.append(PK)

print("PK ACCELERATION",PK_acceleration)


PK_velocity = []

for i in velocity:
    PK = PK_ventaneado(window, i, sampling_rate)
    PK_velocity.append(PK)

print("PK VELOCITY",PK_velocity)

PK_displacement = []

for i in acceleration:
    PK = PK_ventaneado(window, i, sampling_rate)
    PK_displacement.append(PK)

print("PK DISPLACEMENT",PK_displacement)

###### KTS

KTS_acceleration=[]

for i in acceleration:
    KTS= KTS_ventaneado(window, i, sampling_rate)
    KTS_acceleration.append(KTS)

print("KTS ACCELERATION",KTS_acceleration)


KTS_velocity = []

for i in velocity:
    KTS = KTS_ventaneado(window, i, sampling_rate)
    KTS_velocity.append(KTS)

print("KTS VELOCITY",KTS_velocity)

KTS_displacement = []

for i in acceleration:
    KTS = KTS_ventaneado(window, i, sampling_rate)
    KTS_displacement.append(KTS)

print("KTS DISPLACEMENT",KTS_displacement)

#####CREST


crest_acc = crest_factor(RMS_acceleration,PK_acceleration)
print("CREST ACCELERATION",crest_acc)


crest_vel = crest_factor(RMS_velocity,PK_velocity)
print("CREST VELOCITY",crest_vel)


crest_disp = crest_factor(RMS_displacement,PK_displacement)
print("CREST DISPLACEMENT",crest_disp)

###FOURIER PLOTS

# for signal in velocity:
#     Fourier_transform(np.array(signal), sampling_rate)


plt.figure(figsize=(10, 5))
plt.title("Orbitales Datos B sensores 1-0", fontdict={
            'family': 'monospace', 'weight': 'bold', 'size': 10})
plt.grid()
plt.plot(displacement[1], displacement[0])
plt.show()