import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat, savemat

# load the .mat file
mat_data = loadmat('eeg_data.mat')

# note: the file contains
# data -> EEG segments (each segment is 4096 samples)
# labels -> class of each segment (0, 1, or 2)
# fs -> sampling frequency (173.61 Hz)

# extract the variables
data = mat_data['data']  # shape: (500, 4096) -> 500 segments each with 4096 samples
labels = mat_data['labels'].flatten() # flatten to 1D array
fs = float(mat_data['fs'][0, 0])  # sampling rate = 173.61 Hz

# after loading the data, if more than 4096 samples, trim to exactly 4096
# noticed that some matlab files stored 1 or more extra samples by accident so we trim them
if data.shape[1] >= 4097:
    print(f"\nNote: Data has {data.shape[1]} samples per segment. Trimmed to exactly 4096")
    data = data[:, :4096]  # keep only first 4096 samples
    print(f"New data shape: {data.shape}")

# print the dataset information for verification
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Sampling rate: {fs} Hz")
print(f"Number of samples per segment: {data.shape[1]}")
print(f"Duration per segment: {data.shape[1]/fs:.2f} seconds")
print(f"\nClass distribution:")
print(f"  Class 0 (Rest): {np.sum(labels == 0)} segments")
print(f"  Class 1 (Active): {np.sum(labels == 1)} segments")
print(f"  Class 2 (Seizure): {np.sum(labels == 2)} segments")

# design and apply 50 Hz notch filter
 # applying the notch filter to remove specific frequency interference
    #parameters:
        # signal_data: Input signal
        # fs: Sampling frequency
        # notch_freq: frequency to remove (here, 50 Hz for electrical interference)
        # quality_factor: if higher then narrower notch
    # and it returns the filtered signal

def apply_notch_filter(signal_data, fs, notch_freq = 50.0, quality_factor = 30):

    # design the notch filter
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    
    # apply filter to each segment
    if signal_data.ndim == 1:
        # if input is only a single segment (1D array/vector))
        filtered = signal.filtfilt(b, a, signal_data)
    else:
        # if input is multiple segments (2D array/matrix)
        filtered = np.zeros_like(signal_data)
        for i in range(signal_data.shape[0]):
            filtered[i, :] = signal.filtfilt(b, a, signal_data[i, :])
    
    return filtered

# apply notch filter to all data (EEG segments)
print("\nApplying 50 Hz notch filter")
data_filtered = apply_notch_filter(data, fs, notch_freq=50.0, quality_factor=30)
print("Signal is now filtered")

# plot the first Segment (Before and After Filtering)
# Get the first EEG segment
first_segment_original = data[0, :]
first_segment_filtered = data_filtered[0, :]

# create time axis
time_axis = np.arange(len(first_segment_original)) / fs

# create the plot
plt.figure(figsize=(14, 8))

# plot 1: full original signal
plt.subplot(3, 2, 1)
plt.plot(time_axis, first_segment_original, 'b', linewidth=0.5)
plt.title('Original EEG Signal (First Segment)', fontsize=12, fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# plot 2: full filtered signal
plt.subplot(3, 2, 2)
plt.plot(time_axis, first_segment_filtered, 'r', linewidth=0.5)
plt.title('Filtered EEG Signal (After 50 Hz Notch)', fontsize=12, fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# plot 3: original signal (zoomed to first 2 seconds)
zoom_samples = int(2 * fs)
plt.subplot(3, 2, 3)
plt.plot(time_axis[:zoom_samples], first_segment_original[:zoom_samples], 'b', linewidth=0.8)
plt.title('Original Signal (First 2 seconds)', fontsize=11)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# plot 4: filtered signal (zoomed to first 2 seconds)
plt.subplot(3, 2, 4)
plt.plot(time_axis[:zoom_samples], first_segment_filtered[:zoom_samples], 'r', linewidth=0.8)
plt.title('Filtered Signal (First 2 seconds)', fontsize=11)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# plot 5: frequency spectrum of original signal
plt.subplot(3, 2, 5)
freq_orig, psd_orig = signal.welch(first_segment_original, fs, nperseg=1024)
plt.semilogy(freq_orig, psd_orig, 'b', linewidth=1)
plt.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50 Hz')
plt.title('Power Spectrum - Original', fontsize=11)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, 85])

# plot 6: frequency spectrum of filtered signal
plt.subplot(3, 2, 6)
freq_filt, psd_filt = signal.welch(first_segment_filtered, fs, nperseg=1024)
plt.semilogy(freq_filt, psd_filt, 'r', linewidth=1)
plt.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50 Hz (removed)')
plt.title('Power Spectrum - Filtered', fontsize=11)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, 85])

plt.tight_layout()
plt.savefig('notch_filter_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'notch_filter_comparison.png'")

# save the filtered data
output_data = {
    'data_original': data,
    'data_filtered': data_filtered,
    'labels': labels,
    'fs': fs
}

savemat('preprocessed_eeg_data.mat', output_data)
print("\nPreprocessed data saved as 'preprocessed_eeg_data.mat'")

# verifyng that filter worked
print("verification: Effect of 50 Hz Notch Filter:")

# calculate power at 50 Hz before and after filtering
idx_50hz = np.argmin(np.abs(freq_orig - 50))
power_reduction = (psd_orig[idx_50hz] - psd_filt[idx_50hz]) / psd_orig[idx_50hz] * 100

print(f"Power at 50 Hz reduced by: {power_reduction:.2f}%") # if filtering worked, should be 100% or close to it
print(f"Signal power (original): {np.var(first_segment_original):.2f}")
print(f"Signal power (filtered): {np.var(first_segment_filtered):.2f}")
print("Done")