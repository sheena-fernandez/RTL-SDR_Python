import asyncio
from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import struct
import pyaudio
import sys


sdr = RtlSdr()

# Setup PyAudio
BITRATE = 44100  # 44100 # Number of frames per second/frameset
BUFFER_SIZE = 5462  # 5462 1024
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=BITRATE,
                frames_per_buffer=BUFFER_SIZE,
                output=True)


async def streaming():
    try:
        async for samples in sdr.stream():
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

            stream.write(bits)

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()

        p.terminate()

    await sdr.stop()
    sdr.close()


if __name__ == "__main__":
    if len(sys.argv) > 2 or len(sys.argv) == 1:
        print("Usage: fm_realtime.py <station number in MHz>")
        exit()
    else:
        try:
            float(sys.argv[1])
        except ValueError:
            print("Usage: fm_realtime.py <station number in MHz>")
            exit()

    F_station = float(sys.argv[1]) * 1e6  # 81.3MHz J-Wave Tokyo Sky Tree
    F_offset = 25000
    Fs = 2.4e6  # Sample rate TODO
    center_freq = F_station - F_offset  # TODO
    N = 256*1024  # 256000 # Samples to capture

    # Configure device
    sdr.sample_rate = Fs
    sdr.center_freq = center_freq
    sdr.gain = 'auto'

    # start FM streaming
    loop = asyncio.get_event_loop()
    loop.run_until_complete(streaming())

    loop.close()
