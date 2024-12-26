# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:57:23 2024

@author: sulth

SIF3012 Computational Tutorial 4, QUESTION 1 - by SULTHAN RAFIF UMAR (S2137646/1)
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the time domain
t = np.linspace(0, 1, 500)  # 500 points in the range [0, 1]


def f_1(t):
    return 2 * (np.sin(2 * np.pi * t))


def f_2(t):
    return 2 * (np.cos(2 * np.pi * t))


def f_3(t):
    return 2 * (np.sin(2 * np.pi * t))


def f(t):
    return f_1(t) + f_2(t) + f_3(t)


# Perform Discrete Fourier Transform (DFT)
F = np.fft.fft(f(t))
freqs = np.fft.fftfreq(len(t), d=(t[1] - t[0]))  # Frequency values

# Perform Inverse Discrete Fourier Transform (IDFT)
f_reconstructed = np.fft.ifft(F)

# Plot the original function
plt.figure(1)
plt.plot(t, f(t), label="Original Function")
plt.xlabel("Time, t (s)")
plt.ylabel("f(x)")
plt.xlim(0, 1)
plt.title("Original Function in Time Domain")
plt.legend()
plt.grid()

# Plot the magnitude of the DFT
plt.figure(2)
plt.plot(freqs, np.abs(F), label="Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("$|F(k)|$")
plt.title("Discrete Fourier Transform (Magnitude Spectrum)")
plt.legend()
plt.grid()

# Plot the reconstructed function
plt.figure(3)
plt.plot(t, f_reconstructed.real, label="Reconstructed Function", linestyle="--")
plt.xlabel("Time, t (s)")
plt.ylabel("$f_{reconstructed} (x)$")
plt.xlim(0, 1)
plt.title("Reconstructed Function from IDFT")
plt.legend()
plt.grid()

# Display the plots
plt.show()
