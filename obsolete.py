import numpy as np

from functools import lru_cache
import albumentations as A

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin

def pad_dataset(X, y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    X_padded = []
    y_padded = []

    for label in unique_classes:
        class_indices = np.where(y == label)[0]
        X_class = X[class_indices]
        y_class = y[class_indices]

        X_padded.append(X_class)
        y_padded.append(y_class)

        samples_needed = max_class_count - len(X_class)

        if samples_needed > 0:
            X_padded.append(np.full((samples_needed, X.shape[1]), np.nan))
            y_padded.append(np.full(samples_needed, label))

    # Concatenate all padded data
    X_padded = np.vstack(X_padded)
    y_padded = np.concatenate(y_padded)

    return X_padded, y_padded


def build_class_image_dict(X, y):
    class_image_dict = {}

    # Loop over each class and store non-NaN images
    for label in np.unique(y):
        # Extract the images and their labels where the image is not NaN
        class_images = X[(y == label) & (~np.isnan(X).any(axis=1))]
        class_image_dict[label] = class_images

    return class_image_dict


class AugmentationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, image_shape=(20,20), random_seed=0):
        self.image_shape = image_shape
        self.random_seed = random_seed
        self.augmenter = A.Compose([
            A.Rotate(limit=15, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
        ])
        self.class_image_dict = {}

    def fit(self, X, y=None):
        self.class_image_dict = build_class_image_dict(X, y)
        return self

    def transform(self, X, y=None):
        A.seed(self.random_seed)
        return np.array([self.augment_image(tuple(img.ravel())) if np.isnan(img).any() else img for img in X])

    @lru_cache(maxsize=6000)
    def augment_image(self, image, label):
        base_image = np.random.choice(self.class_image_dict[label])

        augmented_image = self.augmenter(image=base_image)['image']

        return augmented_image.astype(np.float32)


X_padded, y_padded = shuffle(*pad_dataset(X, y), random_state=42)

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('hog_pipeline', Pipeline([
            ('hog', HOGExtractor()),         # HOG feature extraction (no need for flattening here)
            ('scaler', StandardScaler())     # StandardScaler for HOG features
        ])),
        ('umap_pipeline', Pipeline([
            ('flatten', FlattenImage()),     # Flatten the original image
            ('umap', umap.UMAP())            # Apply UMAP
        ]))
    ])),

    # Classifier
    ('xgb', XGBClassifier(eval_metric='mlogloss'))
])

#X_augmented, y_augmented = shuffle(*augment_dataset(X_train, y_train))

# print(X_augmented.shape, y_augmented.shape)

#plot_labels_histogram(y_augmented, label_mapping)
#plot_labels_matrix(X_augmented, y_augmented)

# Create the UMAP instance for dimensionality reduction
#umap_transformer = umap.UMAP(random_state=42)

# Create the XGBoost classifier
#xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
