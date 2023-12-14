
from tensorflow.data import Dataset, AUTOTUNE
import config

def create_dataset(X, Y, batch_size=32, shuffle_buffer_size=None):
    dataset = Dataset.from_tensor_slices((X, Y))

    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

class ArabicDataloader:
    def __init__(self, X_train, y_train, X_val, y_val) -> None:
        print('dataset loader')
        batch_size = config.batch_size
        shuffle_buffer_size = len(X_train)
        self.train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)
        self.val_dataset = create_dataset(X_val, y_val, batch_size=batch_size)
        print('Dataset loaders loaded')