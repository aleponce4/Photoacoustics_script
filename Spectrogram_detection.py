"""
Real-Time Spectrogram Animation for Laser Pulse Detection
"""
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
BUFFER_SECONDS = 2  # Duration of audio data to display in seconds
BUFFER_SIZE = RATE * BUFFER_SECONDS  # Total samples to maintain in the buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize audio data buffer
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.int16)

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
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Frequency (Hz)')

# To set up the x-axis time labels more accurately
time_vec = np.linspace(0, BUFFER_SECONDS, num=BUFFER_SIZE)
_, _, _, im = ax.specgram(audio_buffer, NFFT=1024, Fs=RATE, noverlap=512, cmap='plasma')
fig.colorbar(im, ax=ax, label='Intensity dB')

def update(frame):
    global audio_buffer
    new_audio_data = get_audio_data()
    # Update the audio buffer
    audio_buffer = np.roll(audio_buffer, -len(new_audio_data))
    audio_buffer[-len(new_audio_data):] = new_audio_data
    # Clear the axis to redraw the spectrogram with updated buffer
    ax.clear()
    ax.set_ylim(0, RATE / 2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    _, _, _, im = ax.specgram(audio_buffer, NFFT=1024, Fs=RATE, noverlap=512, cmap='plasma')
    # Update x-axis to show time labels correctly
    ax.set_xlim(0, BUFFER_SECONDS)
    return im,

# Create animation
ani = FuncAnimation(fig, update, interval=30, blit=False)

plt.show()

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()


