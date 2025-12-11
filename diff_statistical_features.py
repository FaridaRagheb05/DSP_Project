import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("="*90)
print("Loading MAT file...")

mat = loadmat("features_eeg_data.mat")

# Load 2D arrays
data_raw = mat["data_filtered"]       # shape: (samples, time)
data_diff = mat["diff_signal"]        # shape: (samples, time)
labels = mat["labels"].ravel()

print("Raw signal shape:", data_raw.shape)
print("Derivative signal shape:", data_diff.shape)

# Convert 2D → 3D (samples, channels=1, points)
data_raw = data_raw[:, None, :]
data_diff = data_diff[:, None, :]

print("After reshape:")
print("Raw:", data_raw.shape)
print("Diff:", data_diff.shape)

# 1. Statistical feature extraction
def extract_stats(data):
    means = np.mean(data, axis=2)
    stds = np.std(data, axis=2)

    z = (data - means[:, :, None]) / (stds[:, :, None] + 1e-12)

    skew = np.mean(z**3, axis=2)
    kurt = np.mean(z**4, axis=2)

    return np.concatenate([means, stds, skew, kurt], axis=1)

# Raw statistical features
features_stat_raw = extract_stats(data_raw)

# Differentiated statistical features
features_stat_diff = extract_stats(data_diff)

# 2. Flatten raw & diff for classification
X_raw = data_raw.reshape(data_raw.shape[0], -1)
X_diff = data_diff.reshape(data_diff.shape[0], -1)

# 3. Split once for fair comparison

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, labels, test_size=0.2, random_state=42, stratify=labels)

X_train_stat_raw, X_test_stat_raw, _, _ = train_test_split(
    features_stat_raw, labels, test_size=0.2, random_state=42, stratify=labels)

X_train_diff, X_test_diff, _, _ = train_test_split(
    X_diff, labels, test_size=0.2, random_state=42, stratify=labels)

X_train_stat_diff, X_test_stat_diff, _, _ = train_test_split(
    features_stat_diff, labels, test_size=0.2, random_state=42, stratify=labels)

# 4. Standardize each feature set
def standardize(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

X_train_raw, X_test_raw = standardize(X_train_raw, X_test_raw)
X_train_stat_raw, X_test_stat_raw = standardize(X_train_stat_raw, X_test_stat_raw)
X_train_diff, X_test_diff = standardize(X_train_diff, X_test_diff)
X_train_stat_diff, X_test_stat_diff = standardize(X_train_stat_diff, X_test_stat_diff)

# 5. Compute KNN accuracy vs k
def knn_accuracy_vs_k(X_train, X_test, y_train, y_test):
    accs = []
    for k in range(1, 11):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, pred))
    return accs

acc_raw = knn_accuracy_vs_k(X_train_raw, X_test_raw, y_train, y_test)
acc_stat_raw = knn_accuracy_vs_k(X_train_stat_raw, X_test_stat_raw, y_train, y_test)
acc_diff = knn_accuracy_vs_k(X_train_diff, X_test_diff, y_train, y_test)
acc_stat_diff = knn_accuracy_vs_k(X_train_stat_diff, X_test_stat_diff, y_train, y_test)

# 6. Plot accuracy curves
plt.figure(figsize=(12, 7))
ks = range(1, 11)

plt.plot(ks, acc_raw, marker="o", label="Raw Signal")
plt.plot(ks, acc_stat_raw, marker="o", label="Statistical Features (Raw)")
plt.plot(ks, acc_diff, marker="o", label="Differentiated Signal")
plt.plot(ks, acc_stat_diff, marker="o", label="Statistical Features (Differentiated)")

plt.xlabel("k (KNN)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k for All Feature Types")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("accuracy_vs_k_full_comparison.png", dpi=300)
print("Saved: accuracy_vs_k_full_comparison.png")

plt.show()
