"""
Preprocess and Filter the Audio
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import hilbert, butter, filtfilt

# Path to audio file
audio_path = 'TestAudio.wav'

# Load audio file with librosa
data, sr = librosa.load(audio_path, sr=None)  # sr=None to maintain original sample rate

######## Preprocess and Filter the Audio
# Function to apply a low-pass filter
def low_pass_filter(data, cutoff_frequency, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Applying the low-pass filter to smooth the signal
smoothed_data = low_pass_filter(data, cutoff_frequency=100, fs=sr)  # Adjust cutoff frequency as needed

# Calculating the amplitude envelope using the Hilbert transform
analytic_signal = hilbert(smoothed_data)
amplitude_envelope = np.abs(analytic_signal)

# Creating time arrays for both sets of data
times = np.arange(len(data)) / sr

# Set up the figure and subplots
plt.figure(figsize=(14, 10))  # Increased figure size for better clarity

# Plot unfiltered data
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(times, data)
plt.title('Unfiltered Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot filtered data (amplitude envelope)
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(times, amplitude_envelope, color='black')
plt.title('Filtered Audio Waveform (Amplitude Envelope)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Display the plots
plt.tight_layout()  # Automatically adjusts subplot params so that the subplot(s) fits in to the figure area
plt.show()


######## Feature Extraction




######## Detect Pulse Trains