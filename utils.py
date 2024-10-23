import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import confusion_matrix

from skimage.feature import hog
from skimage import img_as_ubyte
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
from skimage.transform import warp, AffineTransform, rotate

def plot_labels_matrix(X, y, img_shape=(20, 20)):
    images_per_label = 30
    unique_labels = np.unique(y)
    num_labels = len(unique_labels)

    fig, axes = plt.subplots(num_labels, images_per_label, figsize=(30, 2 * num_labels))

    vmin, vmax = (0, 255) if X.dtype == np.uint8 else (0, 1)

    for i, label in enumerate(unique_labels):
        label_indices = np.where(y == label)[0]
        selected_indices = label_indices[:images_per_label]

        for j, idx in enumerate(selected_indices):
            ax = axes[i, j]
            ax.imshow(X[idx].reshape(*img_shape), cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_labels_histogram(y, label_mapping):
    unique_labels, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(label_mapping.values(), counts, color='fuchsia')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_xticks(unique_labels)
    ax.grid(axis='y')

    ax.bar_label(bars)
    plt.tight_layout()
    plt.show()

def plot_confusion_heatmap(y_true, y_pred, label_mapping):
    conf_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='magma', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticklabels(label_mapping.values())
    ax.set_yticklabels(label_mapping.values())
    plt.tight_layout()
    plt.show()

def plot_confusions(y_true, y_pred, X, label_mapping, num_samples=64, img_shape=(20, 20)):
    confusion_indices = np.where(y_true != y_pred)[0]

    # Sort confusion indices by true labels
    confusion_indices = confusion_indices[np.argsort(y_true[confusion_indices])]

    # Limit the number of samples to display
    confusion_indices = confusion_indices[:num_samples]
    grid_size = int(np.sqrt(num_samples))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(confusion_indices):
            ax.axis('off')
            continue

        image_idx = confusion_indices[idx]
        image = X[image_idx].reshape(*img_shape)
        true_label = label_mapping[y_true[image_idx]]
        predicted_label = label_mapping[y_pred[image_idx]]

        ax.imshow(image, cmap='gray')
        ax.axis('off')

        ax.set_title(f"{true_label}/{predicted_label}")

    plt.tight_layout()
    plt.show()

def plot_scaling(X):
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.xlabel("Mean values")
    plt.hist(means.flatten(), bins=50, color='fuchsia', edgecolor='black')

    plt.subplot(1, 2, 2)
    plt.xlabel("Standard deviations")
    plt.hist(std_devs.flatten(), bins=50, color='fuchsia', edgecolor='black')

    plt.tight_layout()
    plt.show()

def plot_probs_histogram(probs, bins, inflection_point=None, num_ood=85):
    sorted_indices = np.argsort(probs)
    lowest_stddevs = sorted_indices[:num_ood]

    plt.hist(probs, bins=bins, color='fuchsia', alpha=0.5, label='Standard deviations')

    plt.hist(probs[lowest_stddevs], bins=bins, color='black', alpha=0.7, label=f'{num_ood} lowest stddevs')

    if inflection_point:
        plt.axvline(x=inflection_point, color='red', linestyle='--', label='Inflection point')

    plt.xlabel('Standard deviations')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()

def plot_images(X, img_shape=None):
    num_images = X.shape[0]

    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    vmin, vmax = (0, 255) if X.dtype == np.uint8 else (0, 1)

    for idx, ax in enumerate(axes.ravel()):
        if idx >= num_images:
            ax.axis('off')
            continue

        image = X[idx].reshape(*img_shape) if img_shape else X[idx]
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def subtract_background(image, sigma):
    """Subtract the background from an image,
    by blurring using a gaussian filter to determine the threshold point."""
    subtracted = image - gaussian(image, sigma=sigma)

    # Remove everything under the threshold (leftover noise)
    subtracted[subtracted < 0] = 0
    return subtracted

""" Removes noise. scikit-image's total variation filter appears to be a good fit for the noise signature seen
https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise.html
"""
class ImageDenoiser(TransformerMixin, BaseEstimator):
    def __init__(self, shape=None, weight=0.1, sigma=12):
        self.weight = weight
        self.sigma = sigma
        self.shape = shape

    def fit(self, X, y=None):
        return self

    # TODO: Might need rescale_intensity()
    def transform(self, X, y=None):
        if self.shape:
            return np.array([subtract_background(denoise_tv_chambolle(
                image.reshape(*self.shape), weight=self.weight), sigma=self.sigma).reshape(-1) for image in X])
        return np.array([subtract_background(denoise_tv_chambolle(
            image, weight=self.weight), sigma=self.sigma) for image in X])


class HOGExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, pixels_per_cell=(5, 5), cells_per_block=(2, 2), orientations=9):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        hog_features = []
        for image in X:
            hog_feat = hog(image,
                           pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block,
                           orientations=self.orientations,
                           block_norm='L2-Hys', visualize=False)
            hog_features.append(hog_feat)

        return hog_features


def augment_image(class_images, img_shape=(20, 20), mode='wrap'):
    """Augment by drawing one base image from a list of images (presumably from the same class)."""
    base_image = class_images[np.random.randint(0, len(class_images))]
    original_shape = base_image.shape
    base_image = base_image.reshape(*img_shape)

    translation_transform = AffineTransform(translation=(np.random.uniform(-2, 2), np.random.uniform(-2, 2)))
    moved_image = warp(base_image, translation_transform, mode=mode)

    rotated_image = rotate(moved_image, angle=np.random.uniform(-8, 8), mode=mode)

    return img_as_ubyte(rotated_image).reshape(*original_shape)

def augment_dataset(X, y, img_shape=(20, 20)):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    X_augmented = []
    y_augmented = []

    for label in unique_classes:
        class_indices = np.where(y == label)[0]
        class_images = X[(y == label) & (~np.isnan(X).any(axis=1))]
        X_class = X[class_indices]
        y_class = y[class_indices]

        X_augmented.append(X_class)
        y_augmented.append(y_class)

        samples_needed = max_class_count - len(X_class)

        y_augmented.append(np.full(samples_needed, label))

        for i in range(samples_needed):
            augmented_image = augment_image(class_images, img_shape=img_shape)
            X_augmented.append(augmented_image.reshape(1, -1))

    # Concatenate all padded data
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)

    return X_augmented, y_augmented


class FlattenImage(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.reshape(X.shape[0], -1)

