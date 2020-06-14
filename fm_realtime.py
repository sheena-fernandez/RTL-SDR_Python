from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import struct
import matplotlib.pyplot as plt
import pyaudio
import sys


def visualize_signals(x2, x4, x5, Fs_y):
    # Generate plot of shifted signal
    ax1.specgram(x2, NFFT=2048, Fs=Fs)
    ax1.set_title("Shifted signal (x2)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(-Fs/2, Fs/2)
    ax1.set_xlim(0, len(x2)/Fs)

    ax2.specgram(x4, NFFT=2048, Fs=Fs_y)
    ax2.set_title("x4")
    ax2.set_ylim(-Fs_y/2, Fs_y/2)
    ax2.set_xlim(0, len(x4)/Fs_y)

    # Plot the constellation of x4.
    ax3.scatter(np.real(x4[0:50000]), np.imag(
        x4[0:50000]), color="red", alpha=0.05)
    ax3.set_title("Constellation (x4)")
    ax3.set_xlabel("Real")
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylabel("Imaginary")
    ax3.set_ylim(-1.1, 1.1)

    # Plot the PSD of x5
    ax4.psd(x5, NFFT=2048, Fs=Fs_y, color="blue")
    ax4.set_title("Power Spectral Density (x5)")
    ax4.axvspan(0,             15000,         color="red", alpha=0.2)
    ax4.axvspan(19000-500,     19000+500,     color="green", alpha=0.4)
    ax4.axvspan(19000*2-15000, 19000*2+15000, color="orange", alpha=0.2)
    ax4.axvspan(19000*3-1500,  19000*3+1500,  color="blue", alpha=0.2)
    ax4.ticklabel_format(style='plain', axis='y')

    # Update graph to show real time data
    plt.pause(0.0000001)

    # clear graphs to avoid overlapping
    ax3.clear()  # clear constellation graph
    plt.cla()   # clear the rest of graphs


def stream_fm():
    # Get around 0.1s of data
    samples = sdr.read_samples(N)

    # Convert samples to a numpy array
    x1 = np.array(samples).astype("complex64")

    # To mix the data down, generate a digital complex exponential
    # (with the same length as x1) with phase -F_offset/Fs
    fc1 = np.exp(-1.0j*2.0*np.pi * F_offset/Fs*np.arange(len(x1)))
    # Multiply x1 and the digital complex expontential (baseband)
    x2 = x1 * fc1

    # Filter and downsample the signal
    # An FM broadcast signal has  a bandwidth of 200 kHz
    f_bw = 200000
    n_taps = 64
    # Use Remez algorithm to design filter coefficients
    lpf = signal.remez(
        n_taps, [0, f_bw, f_bw+(Fs/2-f_bw)/4, Fs/2], [1, 0], Hz=Fs)
    x3 = signal.lfilter(lpf, 1.0, x2)

    dec_rate = int(Fs / f_bw)
    x4 = x3[0::dec_rate]
    # Calculate the new sampling rate
    Fs_y = Fs/dec_rate

    # Demodulate signal
    # Polar discriminator
    y5 = x4[1:] * np.conj(x4[:-1])
    x5 = np.angle(y5)

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

    # Scale audio to adjust volume
    x7 *= 10000 / np.max(np.abs(x7))

    # Output as raw 16-bit, 1 channel audio
    output_raw = x7.astype("int16")
    bits = struct.pack(('<%dh' % len(output_raw)), *output_raw)
    # print(len(output_raw), Fs_audio)
    # stream.write(bits)
    visualize_signals(x2, x4, x5, Fs_y)

    return x2, x4, x5, Fs_y


if __name__ == "__main__":
    if len(sys.argv) > 2 or len(sys.argv) == 1:
        print("Usage: fm_realtime.py <station number in MHz>")
        exit(1)
    else:
        try:
            float(sys.argv[1])
        except ValueError:
            print("Usage: fm_realtime.py <station number in MHz>")
            exit(1)

    sdr = RtlSdr()

    F_station = float(sys.argv[1]) * 1e6  # 81.3MHz J-Wave Tokyo Sky Tree
    F_offset = 25000
    Fs = 2.4e6  # Sample rate
    center_freq = F_station - F_offset
    N = 256*1024  # 256000 # Samples to capture

    # Configure device
    sdr.sample_rate = Fs
    sdr.center_freq = center_freq
    sdr.gain = 'auto'

    # setup graph
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axs

    # Setup PyAudio
    BITRATE = 44100  # 44100 # Number of frames per second/frameset
    BUFFER_SIZE = 5462  # 5462 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=BITRATE,
                    frames_per_buffer=BUFFER_SIZE,
                    output=True)

    try:
        while True:
            x2, x4, x5, Fs_y = stream_fm()

            # TODO: Graph generation processing time too slow to run together with streaming
            # visualize_signals(x2, x4, x5, Fs_y)

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()

        p.terminate()

        sdr.close()
