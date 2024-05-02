import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import time as pTime

duration = 1
amplitude = 1
fs = 2000


def generate_sin_sample(f0, f1, length, amplitudes):
    total_samples = f1 * length
    w = 2.0 * np.pi * f0 / f1
    k = np.arange(0, total_samples)
    sin = np.sin(k * w), np.arange(0, length, length / total_samples)
    res = amplitudes * sin
    return res


def generate_cos_sample(f0, f1, length, amplitudes):
    total_samples = f1 * length
    w = 2.0 * np.pi * f0 / f1
    k = np.arange(0, total_samples)
    sin = np.cos(k * w), np.arange(0, length, length / total_samples)
    res = amplitudes * sin
    return res


def get_fft(y, fs):  # Дискретное преобразование Фурье
    N = len(y)

    k = np.arange(0, N)
    Ex = np.exp(-1j * 2 * np.pi / N * np.outer(k, k))
    yf = np.dot(y, Ex)

    Y2 = yf * np.conj(yf)  # Квадрат модуля Фурье-образа
    ff = k * fs / N  # Вектор частоты, Гц

    # plt.figure(1)5
    plt.figure(figsize=(15, 10))
    plt.plot(ff, Y2, 'r')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Fourier-image modulus squared')
    plt.show()

    return yf


def get_python_fft(y, fs):
    N = len(y)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * 1 / fs), N // 2)

    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Fourier-image modulus squared')
    plt.show()

    return yf


def get_error(yf_my, yf_python):
    error_real = np.real(yf_my) - np.real(yf_python)
    error_imag = np.imag(yf_my) - np.imag(yf_python)

    xe = np.arange(0, len(error_real))

    figure, axis = plt.subplots(1, 2)

    axis[0].plot(xe, error_real, 'y')
    axis[0].set_xlabel('Frequency, Hz')
    axis[0].set_ylabel('Error')
    axis[0].set_title("error_real")

    axis[1].plot(xe, error_imag, 'm')
    axis[1].set_xlabel('Frequency, Hz')
    axis[1].set_ylabel('Error')
    axis[1].set_title("error_imag")

    plt.show()


def generate_signal(freq, fs, duration, amplitude, signal_type='sin'):
    time = np.arange(0, duration, 1/fs)
    if signal_type == 'sin':
        signal = amplitude * np.sin(2 * np.pi * freq * time)
    elif signal_type == 'cos':
        signal = amplitude * np.cos(2 * np.pi * freq * time)
    return signal, time

def my_fft(signal, fs):
    N = len(signal)
    k = np.arange(0, N)
    Ex = np.exp(-1j * 2 * np.pi / N * np.outer(k, k))
    return np.dot(signal, Ex)

def python_fft(signal, fs):
    return np.fft.fft(signal)

def compare_fft(yf_my, yf_python):
    error = np.linalg.norm(yf_my - yf_python)
    print("Error:", error)

# 3.2
# Generate sin and cos signals
fs = 10000
duration = 1
amplitude = 1

ys10, time = generate_signal(10, fs, duration, amplitude, signal_type='sin')
yc100, time = generate_signal(100, fs, duration, amplitude, signal_type='cos')
y = ys10 + yc100

plt.plot(time, y, 'g')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()

# Calculate FFTs
yf_my = my_fft(y, fs)
yf_python = python_fft(y, fs)

compare_fft(yf_my, yf_python)

# 3.3
# Measure execution time for different sampling frequencies
freqs = np.arange(100, 10000, 100)
time_taken = np.zeros(len(freqs))

for i, fs in enumerate(freqs):
    y, _ = generate_signal(10, fs, duration, amplitude, signal_type='sin')
    start_time = pTime.time()

    yf = my_fft(y, fs)

    time_taken[i] = pTime.time() - start_time

plt.plot(freqs, time_taken, 'g')
plt.xlabel('Sampling Frequency')
plt.ylabel('Execution Time (s)')
plt.show()