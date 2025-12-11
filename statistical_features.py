import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("Loading MAT file...")
mat = loadmat("/home/hoda/dsp/DSP_Project-1/features_eeg_data.mat")

# Load data: prefer data_filtered, fall back to data_windowed
#
if "data_filtered" in mat:
    data = mat["data_filtered"]
    print("Using data_filtered from .mat")
else:
    data = mat["data_windowed"]
    print("Using data_windowed (no bandpass filtering)")

labels = mat["labels"].ravel()


# RAW FEATURES
X_raw = data.reshape(data.shape[0], -1)


# STATISTICAL FEATURES
def compute_stats_features(x):
    if x.ndim == 3:
        n_samples, n_channels, n_points = x.shape
        feats = []
        for i in range(n_samples):
            sample = x[i]      # (channels, points)
            mean  = np.mean(sample, axis=1)
            std   = np.std(sample, axis=1)
            maxv  = np.max(sample, axis=1)
            minv  = np.min(sample, axis=1)
            skew  = np.mean(((sample - mean[:,None])/(std[:,None]+1e-6))**3, axis=1)
            kurt  = np.mean(((sample - mean[:,None])/(std[:,None]+1e-6))**4, axis=1) - 3
            feats.append(np.concatenate([mean, std, maxv, minv, skew, kurt]))
        return np.array(feats)

    elif x.ndim == 2:
        mean = np.mean(x, axis=1)
        std  = np.std(x, axis=1)
        maxv = np.max(x, axis=1)
        minv = np.min(x, axis=1)
        skew = np.mean(((x - mean[:, None]) / (std[:, None] + 1e-6))**3, axis=1)
        kurt = np.mean(((x - mean[:, None]) / (std[:, None] + 1e-6))**4 - 3, axis=1)
        return np.column_stack([mean, std, maxv, minv, skew, kurt])

    else:
        raise ValueError("Data must be 2D or 3D")


print("Extracting statistical features...")
X_stats = compute_stats_features(data)


# Train/test split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_raw, labels, test_size=0.2, random_state=42, stratify=labels
)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_stats, labels, test_size=0.2, random_state=42, stratify=labels
)

# Standardize
sc1 = StandardScaler()
sc2 = StandardScaler()

Xr_train = sc1.fit_transform(Xr_train)
Xr_test = sc1.transform(Xr_test)

Xs_train = sc2.fit_transform(Xs_train)
Xs_test = sc2.transform(Xs_test)


# Evaluate KNN for k=1..10
def evaluate_knn_k_range(X_train, X_test, y_train, y_test):
    accuracies = []
    ks = range(1, 11)
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
    return ks, accuracies


print("\nEvaluating Raw Data...")
ks_raw, acc_raw = evaluate_knn_k_range(Xr_train, Xr_test, yr_train, yr_test)

print("Evaluating Statistical Features...")
ks_stats, acc_stats = evaluate_knn_k_range(Xs_train, Xs_test, ys_train, ys_test)

best_raw_acc = max(acc_raw)
best_stats_acc = max(acc_stats)


# PLOTS

# 1. Accuracy vs K (Raw Data)
plt.figure(figsize=(8,5))
plt.plot(ks_raw, acc_raw, marker='o')
plt.title("KNN Accuracy vs K (Raw EEG )")
plt.xlabel("k"); plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(ks_raw)
plt.savefig("accuracy_raw.png", dpi=300)
plt.close()

# 2. Accuracy vs K (Statistical Features)
plt.figure(figsize=(8,5))
plt.plot(ks_stats, acc_stats, marker='o')
plt.title("KNN Accuracy vs K (Statistical Features)")
plt.xlabel("k"); plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(ks_stats)
plt.savefig("accuracy_stats.png", dpi=300)
plt.close()

# 3. Performance comparison
plt.figure(figsize=(6,5))
plt.bar(["Raw", "Statistical"], [best_raw_acc, best_stats_acc])
plt.ylabel("Best Accuracy")
plt.title("Performance Comparison: Raw vs Statistical Features")
plt.ylim(0,1)
plt.savefig("performance_comparison.png", dpi=300)
plt.close()
