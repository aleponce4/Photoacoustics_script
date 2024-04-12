"""
Spectrogram for Laser Pulse Detection
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display

# Path to audio file
audio_path = 'TestAudio.wav'

# Load the audio file
sampling_rate, data = wavfile.read(audio_path)

# Generate the spectrogram with a focus on contrast and pattern visibility
plt.figure(figsize=(20, 4))  # Adjust figure size to stretch out the time axis
Pxx, freqs, bins, im = plt.specgram(data, Fs=sampling_rate, NFFT=1024, noverlap=512, scale='dB')
plt.title('Spectrogram of TestAudio.wav')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.colorbar(label='Intensity in dB')

# Display the spectrogram
plt.show()


"""
Load and Visualize Audio Data
"""

# Path to audio file
audio_path = 'TestAudio.wav'

# Load audio file with librosa
data, sr = librosa.load(audio_path, sr=None)  

# Plot audio waveform
plt.figure(figsize=(14, 5))
times = np.arange(len(data))/sr
plt.plot(times, data)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

