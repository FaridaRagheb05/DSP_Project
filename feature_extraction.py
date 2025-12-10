from scipy.io import loadmat, savemat
import scipy
import numpy as np

# load the .mat file
mat_raw = loadmat("preprocessed_eeg_data.mat/preprocessed_eeg_data.mat")

# MATLAB stores everything inside an extra struct layer — unpack safely
mat_data = {}
for key in mat_raw:
    if not key.startswith("__"):
        mat_data[key] = mat_raw[key]

# extract the variables correctly
data_full = mat_data["data_filtered"]        # shape: (500, 4096)
labels    = mat_data["labels"].flatten()     # shape: (500,)
fs        = float(mat_data["fs"][0, 0])      # sampling rate

# window lengths in seconds
window_lengths = [5, 10, 15, 20]

# dictionary to store all window feature sets
all_features = {}

for win_sec in window_lengths:

    print("WINDOW:", f"win_{win_sec}_seconds")

    # convert seconds to samples
    Nwin = int(win_sec * fs)

    # slice each segment to window length
    data = data_full[:, :Nwin]
    print("  data_windowed: shape =", data.shape)

    # Statistical features
    mean_vector = np.mean(data, axis=1)
    var_vector  = np.var(data, axis=1)
    skew_vector = scipy.stats.skew(data, axis=1)
    kurt_vector = scipy.stats.kurtosis(data, axis=1)

    stats_features = np.column_stack([mean_vector, var_vector, skew_vector, kurt_vector])
    print("  stats_features: shape =", stats_features.shape)

    # Temporal derivative
    diff_data = np.diff(data, axis=1)
    print("  diff_signal: shape =", diff_data.shape)

    # Statistical features of derivative
    mean_diff = np.mean(diff_data, axis=1)
    var_diff  = np.var(diff_data, axis=1)
    skew_diff = scipy.stats.skew(diff_data, axis=1)
    kurt_diff = scipy.stats.kurtosis(diff_data, axis=1)

    diff_stats_features = np.column_stack([mean_diff, var_diff, skew_diff, kurt_diff])
    print("  diff_stats_features: shape =", diff_stats_features.shape)

    # Frequency features
    delta_power = []
    theta_power = []
    alpha_power = []
    beta_power  = []
    gamma_power = []

    # Frequency axis stays same for all segments
    freqs = np.fft.fftfreq(Nwin, d=1/fs)

    for seg in data:
        fft_vals = np.fft.fft(seg)
        P = np.abs(fft_vals) ** 2  # power spectrum

        delta_power.append(P[(freqs >= 0.5) & (freqs < 4)].sum())
        theta_power.append(P[(freqs >= 4)   & (freqs < 8)].sum())
        alpha_power.append(P[(freqs >= 8)   & (freqs < 13)].sum())
        beta_power.append(P[(freqs >= 13)  & (freqs < 30)].sum())
        gamma_power.append(P[(freqs >= 30) & (freqs < 45)].sum())

    freq_features = np.column_stack([
        delta_power, theta_power, alpha_power, beta_power, gamma_power
    ])
    print("  freq_features: shape =", freq_features.shape)
    print()

    # store inside dictionary
    all_features[f"win_{win_sec}_seconds"] = {
        "data_windowed": data,
        "stats_features": stats_features,
        "diff_signal": diff_data,
        "diff_stats_features": diff_stats_features,
        "freq_features": freq_features
    }

# save everything
output_data = {
    "all_windows": all_features,
    "labels": labels,
    "fs": fs
}

savemat("features_eeg_data_all_windows.mat", output_data)

print("Saved: features_eeg_data_all_windows.mat")
