"""
Real-Time Spectrogram Animation for Laser Pulse Detection
"""
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import logging

# Setup logging
logging.basicConfig(filename='realtime_spectrogram.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sampling rate
CHUNK = 1024  # Size of each audio chunk

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Function to get audio data
def get_audio_data():
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        return audio_data
    except Exception as e:
        logging.error("Error capturing audio: %s", str(e))
        return np.zeros(CHUNK)

# Initialize plot
fig, ax = plt.subplots()
ax.set_ylim(0, RATE / 2)
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')

# Initial call to specgram to set up the plot
_, _, _, im = ax.specgram(np.zeros(CHUNK), NFFT=1024, Fs=RATE, noverlap=512, cmap='plasma')
fig.colorbar(im, ax=ax, label='Intensity dB')

# Update function for animation
def update(frame):
    audio_data = get_audio_data()
    ax.clear()  # Clear the axis to redraw the spectrogram
    ax.set_ylim(0, RATE / 2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    # Directly use the specgram function to plot the new data
    _, _, _, im = ax.specgram(audio_data, NFFT=1024, Fs=RATE, noverlap=512, cmap='plasma')
    return im,

# Create animation
ani = FuncAnimation(fig, update, interval=30)

plt.show()

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()
