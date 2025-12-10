from scipy.io import loadmat, savemat
import statistics
import scipy
import numpy as np 

# Number of samples
N = 4096

# load the .mat file
mat_data = loadmat('preprocessed_eeg_data.mat')

# extract the variables
data = mat_data['data_filtered']  # the raw EEG time-domain signal, shape: (500, 4096) -> 500 segments each with 4096 samples
labels = mat_data['labels'].flatten() # flatten to 1D array
fs = float(mat_data['fs'][0, 0])  # sampling rate = 173.61 Hz

# Statistical features 
mean_vector = np.mean(data, axis=1)
var_vector  = np.var(data, axis=1)
skew_vector = scipy.stats.skew(data, axis=1)
kurt_vector = scipy.stats.kurtosis(data, axis=1)
# Combine statistical features
stats_features = np.column_stack([mean_vector, var_vector, skew_vector, kurt_vector])

# Temporal derivative
diff_data = np.diff(data, axis=1)

# Statistical features for temporal derivative (differential signal)
mean_diff = np.mean(diff_data, axis=1)
var_diff  = np.var(diff_data, axis=1)
skew_diff = scipy.stats.skew(diff_data, axis=1)
kurt_diff = scipy.stats.kurtosis(diff_data, axis=1)
# Combine differential statistical features
diff_stats_features = np.column_stack([mean_diff, var_diff, skew_diff, kurt_diff])

# Frequency bands features using spectral analysis 
delta_power = []
theta_power = []
alpha_power = []
beta_power = []
gamma_power = []
for seg in data:
    # FFT and power spectrum (for more meaningful values i.e. energy instead of amplitude)
    data_freq = np.fft.fft(seg)
    P = np.abs(data_freq)**2

    # frequency axis
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Extract frequency band powers (sum of power in each band)
    delta_power.append(P[(freqs >= 0.5) & (freqs < 4)].sum())
    theta_power.append(P[(freqs >= 4)   & (freqs < 8)].sum())
    alpha_power.append(P[(freqs >= 8)   & (freqs < 13)].sum())
    beta_power.append(P[(freqs >= 13)  & (freqs < 30)].sum())
    gamma_power.append(P[(freqs >= 30) & (freqs < 45)].sum())
# Combine frequency-band features
freq_features = np.column_stack([delta_power, theta_power, alpha_power, beta_power, gamma_power])

# save the feature data
output_data = {
    'data_filtered': data,
    'stats_features': stats_features,
    'diff_signal': diff_data,
    'diff_stats_features': diff_stats_features,
    'freq_features': freq_features,
    'labels': labels,
    'fs': fs
}
savemat('features_eeg_data.mat', output_data)
