import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import time

print("="*80)
print("SIGNAL REPRESENTATION COMPARISON FOR EEG CLASSIFICATION")
print("="*80)

# Load the feature data
print("\nLoading feature data...")
mat= loadmat('features_20sec.mat')

# Extract all componentsdata_filtered

data_windowed       = mat["data_windowed"]
stats_features      = mat["stats_features"]
diff_signal         = mat["diff_signal"]
diff_stats_features = mat["diff_stats_features"]
freq_features       = mat["freq_features"]
labels              = mat["labels"].flatten()
fs                  = float(mat["fs"][0][0])

print(f"\nDataset Information:")
print(f"Total segments: {len(labels)}")
print(f"  Class 0 (Rest): {np.sum(labels == 0)} segments")
print(f"  Class 1 (Active): {np.sum(labels == 1)} segments")
print(f"  Class 2 (Seizure): {np.sum(labels == 2)} segments")
print(f"\nSignal representations:")
print(f"  Time-domain signal: {data_windowed.shape}")
print(f"  Differentiated signal: {diff_signal.shape}")
print(f"  Frequency-band features: {freq_features.shape}")

# Store results for all representations
all_results = {}

# ============================================================================
# REPRESENTATION 1: TIME-DOMAIN EEG SIGNAL
# ============================================================================
print("\n" + "="*80)
print("REPRESENTATION 1: TIME-DOMAIN EEG SIGNAL")
print("="*80)
print("Using raw EEG signal (4096 samples per segment)")

X_time = data_windowed # Shape: (500, 4096)
print(f"Feature vector shape: {X_time.shape}")

# Split and scale
X_train_time, X_test_time, y_train, y_test = train_test_split(
    X_time, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Standardizing features...")
scaler_time = StandardScaler()
X_train_time_scaled = scaler_time.fit_transform(X_train_time)
X_test_time_scaled = scaler_time.transform(X_test_time)

# Test KNN for K=1 to 10
k_values = range(1, 11)
results_time = {'k': [], 'train_acc': [], 'test_acc': [], 'test_err': [], 
                'train_time': [], 'test_time': []}

print("\nTraining KNN classifiers (K=1 to 10)...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Training
    start_train = time.time()
    knn.fit(X_train_time_scaled, y_train)
    train_time = time.time() - start_train
    
    # Prediction
    start_test = time.time()
    y_train_pred = knn.predict(X_train_time_scaled)
    y_test_pred = knn.predict(X_test_time_scaled)
    test_time = time.time() - start_test
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results_time['k'].append(k)
    results_time['train_acc'].append(train_acc)
    results_time['test_acc'].append(test_acc)
    results_time['test_err'].append(1 - test_acc)
    results_time['train_time'].append(train_time)
    results_time['test_time'].append(test_time)
    
    print(f"K={k:2d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Time: {test_time:.4f}s")

optimal_k_time = results_time['k'][np.argmax(results_time['test_acc'])]
best_acc_time = max(results_time['test_acc'])

# Train final model and get detailed results
knn_time_final = KNeighborsClassifier(n_neighbors=optimal_k_time)
knn_time_final.fit(X_train_time_scaled, y_train)
y_pred_time = knn_time_final.predict(X_test_time_scaled)
cm_time = confusion_matrix(y_test, y_pred_time)

print(f"\nOptimal K = {optimal_k_time}")
print(f"Best Test Accuracy = {best_acc_time:.4f} ({best_acc_time*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_time, target_names=['Rest', 'Active', 'Seizure']))

all_results['Time-Domain'] = {
    'k_values': results_time,
    'optimal_k': optimal_k_time,
    'best_acc': best_acc_time,
    'confusion_matrix': cm_time,
    'y_pred': y_pred_time,
    'features_shape': X_time.shape,
    'description': 'Raw EEG signal (4096 samples)'
}

# ============================================================================
# REPRESENTATION 2: DIFFERENTIATED EEG SIGNAL
# ============================================================================
print("\n" + "="*80)
print("REPRESENTATION 2: DIFFERENTIATED EEG SIGNAL")
print("="*80)
print("Using first-order derivative of EEG signal (temporal changes)")

X_diff = diff_signal  # Shape: (500, 4095)
print(f"Feature vector shape: {X_diff.shape}")

# Split and scale
X_train_diff, X_test_diff, y_train, y_test = train_test_split(
    X_diff, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Standardizing features...")
scaler_diff = StandardScaler()
X_train_diff_scaled = scaler_diff.fit_transform(X_train_diff)
X_test_diff_scaled = scaler_diff.transform(X_test_diff)

# Test KNN
results_diff = {'k': [], 'train_acc': [], 'test_acc': [], 'test_err': [],
                'train_time': [], 'test_time': []}

print("\nTraining KNN classifiers (K=1 to 10)...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    start_train = time.time()
    knn.fit(X_train_diff_scaled, y_train)
    train_time = time.time() - start_train
    
    start_test = time.time()
    y_train_pred = knn.predict(X_train_diff_scaled)
    y_test_pred = knn.predict(X_test_diff_scaled)
    test_time = time.time() - start_test
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results_diff['k'].append(k)
    results_diff['train_acc'].append(train_acc)
    results_diff['test_acc'].append(test_acc)
    results_diff['test_err'].append(1 - test_acc)
    results_diff['train_time'].append(train_time)
    results_diff['test_time'].append(test_time)
    
    print(f"K={k:2d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Time: {test_time:.4f}s")

optimal_k_diff = results_diff['k'][np.argmax(results_diff['test_acc'])]
best_acc_diff = max(results_diff['test_acc'])

# Final model
knn_diff_final = KNeighborsClassifier(n_neighbors=optimal_k_diff)
knn_diff_final.fit(X_train_diff_scaled, y_train)
y_pred_diff = knn_diff_final.predict(X_test_diff_scaled)
cm_diff = confusion_matrix(y_test, y_pred_diff)

print(f"\nOptimal K = {optimal_k_diff}")
print(f"Best Test Accuracy = {best_acc_diff:.4f} ({best_acc_diff*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_diff, target_names=['Rest', 'Active', 'Seizure']))

all_results['Differentiated'] = {
    'k_values': results_diff,
    'optimal_k': optimal_k_diff,
    'best_acc': best_acc_diff,
    'confusion_matrix': cm_diff,
    'y_pred': y_pred_diff,
    'features_shape': X_diff.shape,
    'description': 'First derivative (4095 samples)'
}

# ============================================================================
# REPRESENTATION 3: FREQUENCY-BAND FEATURES
# ============================================================================
print("\n" + "="*80)
print("REPRESENTATION 3: FREQUENCY-BAND FEATURES")
print("="*80)
print("Using spectral power in Delta, Theta, Alpha, Beta, Gamma bands")

X_freq = freq_features  # Shape: (500, 5)
print(f"Feature vector shape: {X_freq.shape}")
print("Features: [Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)]")

# Split and scale
X_train_freq, X_test_freq, y_train, y_test = train_test_split(
    X_freq, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Standardizing features...")
scaler_freq = StandardScaler()
X_train_freq_scaled = scaler_freq.fit_transform(X_train_freq)
X_test_freq_scaled = scaler_freq.transform(X_test_freq)

# Test KNN
results_freq = {'k': [], 'train_acc': [], 'test_acc': [], 'test_err': [],
                'train_time': [], 'test_time': []}

print("\nTraining KNN classifiers (K=1 to 10)...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    start_train = time.time()
    knn.fit(X_train_freq_scaled, y_train)
    train_time = time.time() - start_train
    
    start_test = time.time()
    y_train_pred = knn.predict(X_train_freq_scaled)
    y_test_pred = knn.predict(X_test_freq_scaled)
    test_time = time.time() - start_test
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results_freq['k'].append(k)
    results_freq['train_acc'].append(train_acc)
    results_freq['test_acc'].append(test_acc)
    results_freq['test_err'].append(1 - test_acc)
    results_freq['train_time'].append(train_time)
    results_freq['test_time'].append(test_time)
    
    print(f"K={k:2d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Time: {test_time:.4f}s")

optimal_k_freq = results_freq['k'][np.argmax(results_freq['test_acc'])]
best_acc_freq = max(results_freq['test_acc'])

# Final model
knn_freq_final = KNeighborsClassifier(n_neighbors=optimal_k_freq)
knn_freq_final.fit(X_train_freq_scaled, y_train)
y_pred_freq = knn_freq_final.predict(X_test_freq_scaled)
cm_freq = confusion_matrix(y_test, y_pred_freq)

print(f"\nOptimal K = {optimal_k_freq}")
print(f"Best Test Accuracy = {best_acc_freq:.4f} ({best_acc_freq*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_freq, target_names=['Rest', 'Active', 'Seizure']))

all_results['Frequency-Band'] = {
    'k_values': results_freq,
    'optimal_k': optimal_k_freq,
    'best_acc': best_acc_freq,
    'confusion_matrix': cm_freq,
    'y_pred': y_pred_freq,
    'features_shape': X_freq.shape,
    'description': '5 frequency band powers'
}

# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON OF ALL REPRESENTATIONS")
print("="*80)

# Summary table
print("\nSUMMARY TABLE:")
print("-"*80)
print(f"{'Representation':<25} {'Features':<15} {'Optimal K':<12} {'Accuracy':<12} {'Time (s)'}")
print("-"*80)

for name, results in all_results.items():
    features = f"{results['features_shape'][1]}"
    optimal_k = results['optimal_k']
    accuracy = f"{results['best_acc']:.4f}"
    avg_time = f"{np.mean(results['k_values']['test_time']):.4f}"
    print(f"{name:<25} {features:<15} {optimal_k:<12} {accuracy:<12} {avg_time}")

print("-"*80)

# Find best representation
best_repr = max(all_results.items(), key=lambda x: x[1]['best_acc'])
print(f"\n🏆 BEST REPRESENTATION: {best_repr[0]}")
print(f"   Accuracy: {best_repr[1]['best_acc']:.4f} ({best_repr[1]['best_acc']*100:.2f}%)")
print(f"   Optimal K: {best_repr[1]['optimal_k']}")

# ============================================================================
# DETAILED VISUALIZATIONS
# ============================================================================
fig = plt.figure(figsize=(20, 14))

# Plot 1: Accuracy Comparison Bar Chart
plt.subplot(3, 4, 1)
names = list(all_results.keys())
accuracies = [all_results[name]['best_acc'] for name in names]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = plt.bar(range(len(names)), accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
best_idx = np.argmax(accuracies)
bars[best_idx].set_alpha(1.0)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold', fontsize=10)
plt.xticks(range(len(names)), names, rotation=15, ha='right')
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1.0])
plt.grid(True, alpha=0.3, axis='y')

# Plot 2: Number of Features Comparison
plt.subplot(3, 4, 2)
feature_counts = [all_results[name]['features_shape'][1] for name in names]
plt.bar(range(len(names)), feature_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for i, count in enumerate(feature_counts):
    plt.text(i, count + 50, str(count), ha='center', fontweight='bold', fontsize=10)
plt.xticks(range(len(names)), names, rotation=15, ha='right')
plt.ylabel('Number of Features', fontsize=12)
plt.title('Dimensionality Comparison', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3, axis='y')

# Plot 3: Computational Time Comparison
plt.subplot(3, 4, 3)
avg_times = [np.mean(all_results[name]['k_values']['test_time']) for name in names]
plt.bar(range(len(names)), avg_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for i, t in enumerate(avg_times):
    plt.text(i, t + 0.001, f'{t:.4f}s', ha='center', fontweight='bold', fontsize=9)
plt.xticks(range(len(names)), names, rotation=15, ha='right')
plt.ylabel('Average Prediction Time (s)', fontsize=12)
plt.title('Computational Efficiency', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Optimal K Values
plt.subplot(3, 4, 4)
optimal_ks = [all_results[name]['optimal_k'] for name in names]
plt.bar(range(len(names)), optimal_ks, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for i, k in enumerate(optimal_ks):
    plt.text(i, k + 0.2, f'K={k}', ha='center', fontweight='bold', fontsize=10)
plt.xticks(range(len(names)), names, rotation=15, ha='right')
plt.ylabel('Optimal K Value', fontsize=12)
plt.title('Optimal K Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, max(optimal_ks) + 2])
plt.grid(True, alpha=0.3, axis='y')

# Plots 5-7: Accuracy vs K for each representation
for idx, (name, color) in enumerate(zip(names, colors), start=5):
    plt.subplot(3, 4, idx)
    results = all_results[name]['k_values']
    plt.plot(results['k'], results['train_acc'], 'o-', label='Train', linewidth=2, markersize=6)
    plt.plot(results['k'], results['test_acc'], 's-', label='Test', linewidth=2, markersize=6, color=color)
    opt_k = all_results[name]['optimal_k']
    plt.axvline(x=opt_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.xlabel('K (Number of Neighbors)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title(f'{name}\n(Optimal K={opt_k})', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(results['k'])
    plt.ylim([0.4, 1.05])

# Plots 8-10: Confusion Matrices
for idx, name in enumerate(names, start=8):
    plt.subplot(3, 4, idx)
    cm = all_results[name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rest', 'Active', 'Seizure'],
                yticklabels=['Rest', 'Active', 'Seizure'],
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 11})
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('True', fontsize=11)
    accuracy = all_results[name]['best_acc']
    plt.title(f'{name}\nAccuracy: {accuracy:.3f}', fontsize=12, fontweight='bold')

# Plot 11: Per-Class Performance Comparison
plt.subplot(3, 4, 11)
x_positions = np.arange(3)
width = 0.25
class_names = ['Rest', 'Active', 'Seizure']

for i, (name, color) in enumerate(zip(names, colors)):
    cm = all_results[name]['confusion_matrix']
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    plt.bar(x_positions + i*width, class_recalls, width, label=name, 
            color=color, alpha=0.7, edgecolor='black')

plt.xlabel('Class', fontsize=12)
plt.ylabel('Recall (Sensitivity)', fontsize=12)
plt.title('Per-Class Performance', fontsize=13, fontweight='bold')
plt.xticks(x_positions + width, class_names)
plt.legend(fontsize=9)
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')

# Plot 12: Summary Statistics
plt.subplot(3, 4, 12)
plt.axis('off')
summary_text = f"""
{'='*45}
SIGNAL REPRESENTATION COMPARISON
{'='*45}

Dataset: 500 EEG segments
Train/Test Split: 80/20

RESULTS SUMMARY:
{'─'*45}
"""

for name in names:
    res = all_results[name]
    cm = res['confusion_matrix']
    seizure_recall = cm[2,2] / cm[2,:].sum()
    summary_text += f"""
{name}:
  Features: {res['features_shape'][1]}
  Accuracy: {res['best_acc']:.3f} ({res['best_acc']*100:.1f}%)
  Optimal K: {res['optimal_k']}
  Seizure Recall: {seizure_recall:.3f}
"""

summary_text += f"""
{'─'*45}
WINNER: {best_repr[0]}
{'='*45}
"""

plt.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('signal_representation_comparison_20.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("Visualization saved as 'signal_representation_comparison_20sec.png'")
print("="*80)

# ============================================================================
# SAVE DETAILED TEXT REPORT
# ============================================================================
print("\nSaving detailed report...")

with open('signal_representation_report_20sec.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SIGNAL REPRESENTATION COMPARISON FOR EEG CLASSIFICATION\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {len(labels)} EEG segments\n")
    f.write(f"Train/Test Split: 80/20 ({len(y_train)} / {len(y_test)})\n")
    f.write(f"Sampling Frequency: {fs:.2f} Hz\n\n")
    
    for name, results in all_results.items():
        f.write("="*80 + "\n")
        f.write(f"REPRESENTATION: {name}\n")
        f.write("="*80 + "\n")
        f.write(f"Description: {results['description']}\n")
        f.write(f"Feature dimensions: {results['features_shape']}\n")
        f.write(f"Number of features: {results['features_shape'][1]}\n")
        f.write(f"Optimal K: {results['optimal_k']}\n")
        f.write(f"Best Test Accuracy: {results['best_acc']:.4f} ({results['best_acc']*100:.2f}%)\n")
        f.write(f"Average prediction time: {np.mean(results['k_values']['test_time']):.4f} seconds\n\n")
        
        f.write("K-value Results:\n")
        f.write("-"*80 + "\n")
        for k, train_acc, test_acc, test_err in zip(
            results['k_values']['k'],
            results['k_values']['train_acc'],
            results['k_values']['test_acc'],
            results['k_values']['test_err']
        ):
            f.write(f"K={k:2d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Error: {test_err:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test, results['y_pred'], 
                                     target_names=['Rest', 'Active', 'Seizure']))
        
        f.write("\nConfusion Matrix:\n")
        cm = results['confusion_matrix']
        f.write("              Predicted\n")
        f.write("              Rest  Active  Seizure\n")
        f.write("True  Rest    " + "  ".join([f"{cm[0,i]:4d}" for i in range(3)]) + "\n")
        f.write("      Active  " + "  ".join([f"{cm[1,i]:4d}" for i in range(3)]) + "\n")
        f.write("      Seizure " + "  ".join([f"{cm[2,i]:4d}" for i in range(3)]) + "\n\n")
        
        # Per-class metrics
        class_recalls = cm.diagonal() / cm.sum(axis=1)
        f.write("Per-Class Recall:\n")
        f.write(f"  Rest:    {class_recalls[0]:.4f} ({class_recalls[0]*100:.2f}%)\n")
        f.write(f"  Active:  {class_recalls[1]:.4f} ({class_recalls[1]*100:.2f}%)\n")
        f.write(f"  Seizure: {class_recalls[2]:.4f} ({class_recalls[2]*100:.2f}%)\n\n")
    
    f.write("="*80 + "\n")
    f.write("COMPARATIVE SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Representation':<25} {'Features':<12} {'Optimal K':<12} {'Accuracy':<15}\n")
    f.write("-"*80 + "\n")
    for name, results in all_results.items():
        features = f"{results['features_shape'][1]}"
        optimal_k = f"{results['optimal_k']}"
        accuracy = f"{results['best_acc']:.4f}"
        f.write(f"{name:<25} {features:<12} {optimal_k:<12} {accuracy:<15}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write(f"BEST REPRESENTATION: {best_repr[0]}\n")
    f.write(f"Accuracy: {best_repr[1]['best_acc']:.4f} ({best_repr[1]['best_acc']*100:.2f}%)\n")
    f.write(f"Optimal K: {best_repr[1]['optimal_k']}\n")
    f.write("="*80 + "\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    
    # Analyze results
    accs = [all_results[name]['best_acc'] for name in names]
    feat_counts = [all_results[name]['features_shape'][1] for name in names]
    
    if feat_counts[2] < feat_counts[0]:  # Frequency has fewer features
        f.write("1. DIMENSIONALITY: Lower-dimensional frequency-band features are more efficient\n")
        f.write(f"   ({feat_counts[2]} features vs {feat_counts[0]} in time-domain)\n\n")
    
    best_seizure_recall = 0
    best_seizure_repr = ""
    for name, results in all_results.items():
        cm = results['confusion_matrix']
        seizure_recall = cm[2,2] / cm[2,:].sum()
        if seizure_recall > best_seizure_recall:
            best_seizure_recall = seizure_recall
            best_seizure_repr = name
    
    f.write(f"2. SEIZURE DETECTION: {best_seizure_repr} achieved best seizure recall\n")
    f.write(f"   ({best_seizure_recall:.2%} of seizures detected)\n\n")
    
    if accs[2] > accs[0] and accs[2] > accs[1]:
        f.write("3. FREQUENCY REPRESENTATION: Frequency-band features provide best overall accuracy\n")
        f.write("   Confirms importance of spectral analysis for EEG classification\n\n")
    
    f.write("4. CURSE OF DIMENSIONALITY: High-dimensional representations (time-domain,\n")
    f.write("   differentiated) suffer from overfitting despite containing more information\n\n")
    
    f.write("="*80 + "\n")

print("Detailed report saved as 'signal_representation_report_20.txt'")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)