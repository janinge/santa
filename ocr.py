import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def plot_labels_matrix(X, y):
    images_per_label = 30
    unique_labels = np.unique(y)
    num_labels = len(unique_labels)

    fig, axes = plt.subplots(num_labels, images_per_label, figsize=(30, 2 * num_labels))

    for i, label in enumerate(unique_labels):
        label_indices = np.where(y == label)[0]
        selected_indices = label_indices[:images_per_label]

        for j, idx in enumerate(selected_indices):
            ax = axes[i, j]
            ax.imshow(X[idx].reshape(20, 20), cmap='gray', vmin=0, vmax=255)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_labels_histogram(X, y, label_mapping):
    unique_labels, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(label_mapping.values(), counts, color='fuchsia')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_xticks(unique_labels)
    ax.grid(axis='y')

    ax.bar_label(bars)
    plt.show()

dataset = np.load("data/dataset.npz")
X, y = dataset["X"], dataset["y"]

label_mapping = {i: hex(i)[2:].upper() for i in range(16)}
label_mapping[16] = "blank"

# First we inspect some of the images visually, and check the label distribution in the dataset
#plot_labels_histogram(X, y, label_mapping)
#plot_labels_matrix(X, y)

# Keep just 10% for testing, since the dataset is rather large
# Stratify tries to keep the same distribution of labels in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Set up a pipeline: PCA > SVM
pipeline = Pipeline([
    ('pca', PCA(n_components=60)),
    ('svm', SVC(kernel='rbf', C=1.0))
])

# And train it
pipeline.fit(X_train, y_train)

# Get a baseline accuracy score that we can try to improve on with a more complex pipeline (including hyperparameter tuning)
accuracy = pipeline.score(X_test, y_test)

print(accuracy)
