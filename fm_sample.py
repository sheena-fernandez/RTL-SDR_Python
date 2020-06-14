from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


# get device
sdr = RtlSdr()

F_station = 81.3e6  # 81.3MHz J-Wave Tokyo Sky Tree
F_offset = 25000   # Offset to capture at (to avoid DC spike)
center_freq = F_station - F_offset  # Center frequency
Fs = int(1140000)   # Sample rate
N = int(8192000)    # Samples to capture
# N = int(8192000*2)    # Samples to capture

# configure the device
sdr.sample_rate = Fs      # Hz
sdr.center_freq = center_freq      # Hz
sdr.gain = 'auto'

# Read specified number of complex samples from tuner.
# Real and imaginary parts are normalized in the range [-1,1]
samples = sdr.read_samples(N)

# Clean up the SDR device
sdr.close()
del(sdr)

# Convert samples to a numpy array
x1 = np.array(samples).astype("complex64")
# Plot spectogram
plt.specgram(x1, NFFT=2048, Fs=Fs)
plt.title("Samples spectogram (x1)")
plt.ylim(-Fs/2, Fs/2)
plt.savefig("x1_spec.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

# To mix the data down, generate a digital complex exponential
# (with the same length as x1) with phase -F_offset/Fs
fc1 = np.exp(-1.0j*2.0*np.pi * F_offset/Fs*np.arange(len(x1)))
# Multiply x1 and the digital complex expontential (baseband)
x2 = x1 * fc1

# Generate plot of shifted signal
plt.specgram(x2, NFFT=2048, Fs=Fs)
plt.title("Shifted signal (x2)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(-Fs/2, Fs/2)
plt.xlim(0, len(x2)/Fs)
plt.ticklabel_format(style='plain', axis='y')
plt.savefig("x2_spec.png", bbox_inches='tight', pad_inches=0.5)
plt.close()


# Filter and downsample the signal
# An FM broadcast signal has  a bandwidth of 200 kHz
f_bw = 200000
n_taps = 64
# Use Remez algorithm to design filter coefficients
lpf = signal.remez(n_taps, [0, f_bw, f_bw+(Fs/2-f_bw)/4, Fs/2], [1, 0], Hz=Fs)
x3 = signal.lfilter(lpf, 1.0, x2)

dec_rate = int(Fs / f_bw)
x4 = x3[0::dec_rate]
# Calculate the new sampling rate
Fs_y = Fs/dec_rate

plt.specgram(x4, NFFT=2048, Fs=Fs_y)
plt.title("x4")
plt.ylim(-Fs_y/2, Fs_y/2)
plt.xlim(0, len(x4)/Fs_y)
plt.ticklabel_format(style='plain', axis='y')
plt.savefig("x4_spec.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

# Plot the constellation of x4.
plt.scatter(np.real(x4[0:50000]), np.imag(
    x4[0:50000]), color="red", alpha=0.05)
plt.title("Constellation (x4)")
plt.xlabel("Real")
plt.xlim(-1.1, 1.1)
plt.ylabel("Imag")
plt.ylim(-1.1, 1.1)
plt.savefig("x4_const.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

# Demodulate signal
# Polar discriminator
y5 = x4[1:] * np.conj(x4[:-1])
x5 = np.angle(y5)

# x5 is an array of real values
# As a result, the PSDs will now be plotted single-sided by default (since
# a real signal has a symmetric spectrum)
# Plot the PSD of x5
plt.psd(x5, NFFT=2048, Fs=Fs_y, color="blue")
plt.title("Power Spectral Density (x5)")
plt.axvspan(0,             15000,         color="red", alpha=0.2)
plt.axvspan(19000-500,     19000+500,     color="green", alpha=0.4)
plt.axvspan(19000*2-15000, 19000*2+15000, color="orange", alpha=0.2)
plt.axvspan(19000*3-1500,  19000*3+1500,  color="blue", alpha=0.2)
plt.ticklabel_format(style='plain', axis='y')
plt.savefig("x5_psd.png", bbox_inches='tight', pad_inches=0.5)
plt.close()


# De-emphasis filter
# Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
d = Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point
x = np.exp(-1/d)   # Calculate the decay between each sample
b = [1-x]          # Create the filter coefficients
a = [1, -x]
x6 = signal.lfilter(b, a, x5)


# Find a decimation rate to achieve audio sampling rate between 44-48 kHz
audio_freq = 44100.0
dec_audio = int(Fs_y/audio_freq)
Fs_audio = Fs_y / dec_audio

x7 = signal.decimate(x6, dec_audio)


# Write to an audio file
# Scale audio to adjust volume
x7 *= 10000 / np.max(np.abs(x7))
# Save to file as 16-bit signed single-channel audio samples
x7.astype("int16").tofile("wbfm-mono.raw")

# Print audio sampling rate
print(Fs_audio)
