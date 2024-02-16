"""
Title: Real-Time Audio Monitoring for Photoacostuic detection
Description: This script captures and processes audio data in real-time to laser pulse train sounds. It visualizes audio waveforms, performs FFT analysis, and detects specific frequency patterns.
Author: Alejandro Ponce - aleponce92@gmail.com
Dependencies: pyaudio, numpy, scipy, matplotlib
"""

# Import Libraries
import pyaudio
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# Setup logging
logging.basicConfig(filename='audio_monitoring.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Parameter Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
CENTER_FREQUENCY = 500
HALF_WIDTH = 150
FREQUENCY_RANGE = (CENTER_FREQUENCY - HALF_WIDTH, CENTER_FREQUENCY + HALF_WIDTH)
amplitude_threshold = 100000
# Expected laser pattern
EXPECTED_LENGTH = 4
MIN_LENGTH = 3
MAX_ZERO_GAP = 2

# Initialization
audio = pyaudio.PyAudio()

# Log available microphones
try:
    available_mics = [
        audio.get_device_info_by_index(i)['name']
        for i in range(audio.get_device_count())
        if audio.get_device_info_by_index(i)['maxInputChannels'] > 0
    ]
    logging.info("Available microphones: %s", available_mics)
except Exception as e:
    logging.error("Failed to list available microphones: %s", str(e))

# Audio Capture Function
def capture_audio(chunk=CHUNK, device_index=None):   # Chane Microhpone device here
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        frames_per_buffer=chunk, input_device_index=device_index)
    data = stream.read(chunk, exception_on_overflow=False)  # Prevent overflow exceptions
    stream.stop_stream()
    stream.close()
    return np.frombuffer(data, dtype=np.int16)

# FFT Function
def fft_and_filter(audio_data):
    yf = fft(audio_data)
    xf = fftfreq(len(audio_data), 1 / RATE)
    return xf, yf

# Visualization Update Functions
def update_waveform(frame):
    audio_data = capture_audio()
    line_waveform.set_ydata(audio_data)
    return line_waveform,

def update_fft(frame):
    audio_data = capture_audio()
    xf, yf = fft_and_filter(audio_data)
    xf = xf[:len(xf)//2]  # Keep only the positive frequencies
    yf = np.abs(yf[:len(yf)//2])
    line_fft.set_data(xf, yf)
    # Reset previously set lines to avoid duplication
    for line in axs[1].lines[1:]:
        line.set_data([], [])
    # threshold lines
    axs[1].axvline(x=FREQUENCY_RANGE[0], color='r', linestyle='--')
    axs[1].axvline(x=FREQUENCY_RANGE[1], color='r', linestyle='--')
    axs[1].axhline(y=amplitude_threshold, color='g', linestyle='--')
    axs[1].relim()
    axs[1].autoscale_view()
    return line_fft,


binary_history_length = 100  # Number of points to display in the binary signal plot
# Initialize a list or array to keep track of the binary signal presence history
binary_signal_presence_history = np.zeros(binary_history_length, dtype=int)

def update_time_resolved(frame):
    global binary_signal_presence_history
    
    audio_data = capture_audio()
    xf, yf = fft_and_filter(audio_data)
    xf = xf[:len(xf)//2]  # Keep only the positive frequencies
    yf = np.abs(yf[:len(yf)//2])  # Get the magnitude of the positive frequencies

    # Filtering based on frequency range
    mask = (xf >= FREQUENCY_RANGE[0]) & (xf <= FREQUENCY_RANGE[1])
    yf_filtered = yf[mask]
    # Check if any frequency component within the range exceeds the amplitude threshold
    signal_presence = np.any(yf_filtered >= amplitude_threshold).astype(int)

    # Update the binary signal presence history
    binary_signal_presence_history = np.roll(binary_signal_presence_history, -1)
    binary_signal_presence_history[-1] = signal_presence
    
    # Update the plot with the binary signal presence history
    line_time_resolved.set_data(np.arange(len(binary_signal_presence_history)), binary_signal_presence_history)
    
    # Ensure the plot is redrawn with the new data
    axs[2].relim()  # Recompute the data limits
    axs[2].autoscale_view()  # Rescale the view limits

    return line_time_resolved,

def detect_pattern_breaks(binary_signal):
    in_train = False  # Are we currently in a train of 1s?
    train_length = 0  # Length of the current train of 1s
    gap_length = 0  # Length of the current gap of 0s

    for bit in binary_signal:
        if bit == 1:
            if not in_train:
                # Starting a new train, check the gap before it
                if gap_length > MAX_ZERO_GAP:
                    print("Unexpected long gap detected.")
                in_train = True
                gap_length = 0  # Reset gap length
            train_length += 1
        else:  # bit == 0
            if in_train:
                # Exiting a train, check its length
                if train_length < MIN_LENGTH:
                    print("Missing or short train detected.")
                in_train = False
                train_length = 0  # Reset train length
            gap_length += 1

    # Check the last train if we ended in one
    if in_train and train_length < MIN_LENGTH:
        print("Missing or short train detected at the end.")



# Setup for Visualization
fig, axs = plt.subplots(3, figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)

# Plot configurations
axs[0].set_title('Live Audio Waveform')
axs[0].set_xlabel('Samples')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlim(0, CHUNK)
axs[0].set_ylim(-32768, 32767)
line_waveform, = axs[0].plot(np.arange(CHUNK), np.zeros(CHUNK), lw=1)

axs[1].set_title('Live FFT of Audio Signal')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')
axs[1].set_xlim(0, RATE / 10)
line_fft, = axs[1].plot([], [], lw=1)

axs[2].set_title('Filtered Signal Presence')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Presence')
axs[2].set_ylim(-0.5, 1.5)  # only once here for initial setup
line_time_resolved, = axs[2].plot([], [], lw=1)


# Animation
ani_waveform = FuncAnimation(fig, update_waveform, blit=True, interval=200, cache_frame_data=False)
ani_fft = FuncAnimation(fig, update_fft, blit=True, interval=200, cache_frame_data=False)
ani_time_resolved = FuncAnimation(fig, update_time_resolved, blit=True, interval=200, cache_frame_data=False)

plt.show()

# Clean up PyAudio
audio.terminate()
