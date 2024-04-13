"""
Preprocess and Filter the Audio
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, savgol_filter, hilbert, filtfilt


# Path to audio file
audio_path = 'TestAudio.wav'

# Load audio file with librosa
data, sr = librosa.load(audio_path, sr=None)  # sr=None to maintain original sample rate

#################### Preprocess and Filter the Audio

#### Step 1
# Define a bandpass and butterworth  filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

# Apply filter
filtered_data = bandpass_filter(data, lowcut=1000, highcut=10000, fs=sr, order=6)

# Creating time arrays for both sets of data
times = np.arange(len(data)) / sr

# Set up the figure and subplots
plt.figure(figsize=(14, 10))  # Increased figure size for better clarity

# Plot unfiltered data
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(times, data, linewidth=0.5)
plt.title('Unfiltered Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot filtered data (amplitude envelope)
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(times, filtered_data, color='black', linewidth=0.5)
plt.title('Filtered Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Display the plots
plt.tight_layout()  # Automatically adjusts subplot params so that the subplot(s) fits in to the figure area
plt.show()


#### Step 2

# Amplitude Normalization
normalized_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)

# Alternatively, apply Savitzky-Golay filter for smoother results
smoothed_data = savgol_filter(normalized_data, window_length=1000, polyorder=3)



# Create the figure and subplots
plt.figure(figsize=(14, 10))  # Increased figure size for better clarity

# Plot the original filtered data
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(np.arange(len(filtered_data)) / sr, filtered_data, label='Filtered Data', color='black', linewidth=0.5)
plt.title('Filtered Audio Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the smoothed data
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(np.arange(len(smoothed_data)) / sr, smoothed_data, label='Smoothed Data', color='red', linewidth=0.5)
plt.title('Smoothed Audio Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()

# Display the plots
plt.tight_layout()  # Automatically adjusts subplot params so that the subplot(s) fits in to the figure area
plt.show()





######## Feature Extraction




######## Detect Pulse Trains