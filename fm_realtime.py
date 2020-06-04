from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import struct
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess

sdr = RtlSdr()

F_station = 81.3e6  # 81.3MHz J-Wave Tokyo Sky Tree
F_offset = 25000
Fs = 2.4e6  # Sample rate
center_freq = F_station - F_offset
N = 256000 # 256*1024 # Samples to capture

# Configure device
sdr.sample_rate = Fs
sdr.center_freq = center_freq
sdr.gain = 'auto'

# setup graph
fig, axs = plt.subplots(2,2)
(ax1, ax2), (ax3, ax4) = axs

command = "sox -t raw -r 256000 -b 16 -c 1 -L -e signed-integer - -d rate 32000"

def fm_sample(i):
    # Get around 0.1s of data
    samples = sdr.read_samples(N)

    # Convert samples to a numpy array
    x1 = np.array(samples).astype("complex64")    

    # To mix the data down, generate a digital complex exponential 
    # (with the same length as x1) with phase -F_offset/Fs
    fc1 = np.exp(-1.0j*2.0*np.pi* F_offset/Fs*np.arange(len(x1)))  
    # Multiply x1 and the digital complex expontential (baseband)
    x2 = x1 * fc1

    ## Generate plot of shifted signal
    ax1.specgram(x2, NFFT=2048, Fs=Fs)  
    ax1.set_title("Shifted signal (x2)")  
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel("Frequency (Hz)")  
    ax1.set_ylim(-Fs/2, Fs/2)  
    ax1.set_xlim(0,len(x2)/Fs)  
    # ax1.set_ticklabel_format(style='plain', axis='y' )  

    ## Filter and downsample the signal
    # An FM broadcast signal has  a bandwidth of 200 kHz
    f_bw = 200000  
    n_taps = 64
    # Use Remez algorithm to design filter coefficients
    lpf = signal.remez(n_taps, [0, f_bw, f_bw+(Fs/2-f_bw)/4, Fs/2], [1,0], Hz=Fs)  
    x3 = signal.lfilter(lpf, 1.0, x2)

    dec_rate = int(Fs / f_bw)  
    x4 = x3[0::dec_rate]  
    # Calculate the new sampling rate
    Fs_y = Fs/dec_rate  

    ax2.specgram(x4, NFFT=2048, Fs=Fs_y)  
    ax2.set_title("x4")  
    ax2.set_ylim(-Fs_y/2, Fs_y/2)  
    ax2.set_xlim(0,len(x4)/Fs_y)  
    # ax2.ticklabel_format(style='plain', axis='y' ) 

    ## Plot the constellation of x4.  
    ax3.scatter(np.real(x4[0:50000]), np.imag(x4[0:50000]), color="red", alpha=0.05)  
    ax3.set_title("Constellation (x4)")  
    ax3.set_xlabel("Real")  
    ax3.set_xlim(-1.1,1.1)  
    ax3.set_ylabel("Imaginary")  
    ax3.set_ylim(-1.1,1.1)  

    ## Demodulate signal
    ### Polar discriminator
    y5 = x4[1:] * np.conj(x4[:-1])  
    x5 = np.angle(y5)

    # Plot the PSD of x5
    ax4.psd(x5, NFFT=2048, Fs=Fs_y, color="blue")  
    ax4.set_title("Power Spectral Density (x5)")  
    ax4.axvspan(0,             15000,         color="red", alpha=0.2)  
    ax4.axvspan(19000-500,     19000+500,     color="green", alpha=0.4)  
    ax4.axvspan(19000*2-15000, 19000*2+15000, color="orange", alpha=0.2)  
    ax4.axvspan(19000*3-1500,  19000*3+1500,  color="blue", alpha=0.2)  
    # ax4.ticklabel_format(style='plain', axis='y' )  

    # De-emphasis filter
    # Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
    d = Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point  
    x = np.exp(-1/d)   # Calculate the decay between each sample  
    b = [1-x]          # Create the filter coefficients  
    a = [1,-x]  
    x6 = signal.lfilter(b,a,x5)  

    # Find a decimation rate to achieve audio sampling rate between 44-48 kHz
    audio_freq = 44100.0  
    dec_audio = int(Fs_y/audio_freq)  
    Fs_audio = Fs_y / dec_audio

    x7 = signal.decimate(x6, dec_audio)  

    ## Write to an audio file
    # Scale audio to adjust volume
    x7 *= 10000 / np.max(np.abs(x7))
    # x7.astype("int16").tofile("wbfm-mono.raw")  

    
    # Update graph to show real time data 
    # plt.pause(1)

    # clear graphs to avoid overlapping 
    # plt.cla()

    output_raw = x7.astype(int)

    # # Output as raw 16-bit, 1 channel audio
    bits = struct.pack(('<%dh' % len(output_raw)), *output_raw)
    # bits = struct.pack(('<%dh' % len(x7)), *x7)

    # sys.stdout.buffer.write(bits)

    # subprocess.call(command, shell=True)


ani = FuncAnimation(fig, fm_sample,
                    interval=1000, repeat=False)

plt.show()
sdr.close()