# Import Libraries
import pyaudio
import numpy as np
from scipy.fft import fft, fftfreq
import logging
import datetime

# Setup logging
logging.basicConfig(filename='pattern_characterization.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Parameter Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024  # Consider adjusting for optimal performance
CENTER_FREQUENCY = 500
HALF_WIDTH = 150
FREQUENCY_RANGE = (CENTER_FREQUENCY - HALF_WIDTH, CENTER_FREQUENCY + HALF_WIDTH)
amplitude_threshold = 100000

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
def capture_audio(chunk=CHUNK, device_index=None):
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=chunk, input_device_index=device_index)
        data = stream.read(chunk, exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
        return np.frombuffer(data, dtype=np.int16)
    except Exception as e:
        logging.error("Failed to capture audio: %s", str(e))
        return np.zeros(chunk)

# FFT Function
def fft_and_filter(audio_data):
    yf = fft(audio_data)
    xf = fftfreq(len(audio_data), 1 / RATE)
    mask = (xf >= FREQUENCY_RANGE[0]) & (xf <= FREQUENCY_RANGE[1])
    yf_filtered = yf[mask]
    return np.abs(yf_filtered)

# Main Processing Loop
def process_audio():
    binary_signal = []
    for _ in range(100):  # Adjust based on how long you want to capture and analyze
        audio_data = capture_audio()
        yf_filtered = fft_and_filter(audio_data)
        signal_presence = np.any(yf_filtered >= amplitude_threshold).astype(int)
        binary_signal.append(signal_presence)
    
    return binary_signal

# Save Binary Signal
def save_binary_signal(binary_signal):
    filename = f"binary_signal_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w') as file:
        file.write(' '.join(map(str, binary_signal)))
    logging.info(f"Binary signal saved to {filename}")

if __name__ == "__main__":
    binary_signal = process_audio()
    save_binary_signal(binary_signal)

# Clean up PyAudio
audio.terminate()
