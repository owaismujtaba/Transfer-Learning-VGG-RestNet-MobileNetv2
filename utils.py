
import config
import pandas as pd
from tensorflow import convert_to_tensor, expand_dims, 
from tensorflow.image  import grayscale_to_rgb, resize


def load_AHWD_dataset():
    print('loading ahwd dataset')
    X_train_path = config.X_train_path
    X_test_path = config.X_test_path

    y_train_path = config.y_train_path
    y_test_path = config.y_test_path

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    y_train = y_train-1
    y_test = y_test - 1
    return X_train.values, y_train.values,  X_test.values, y_test.values



def load_processed_dataset():
    X_train, y_train, X_val, y_val = load_AHWD_dataset()
    
    X_train = X_train.reshape(-1, 32, 32)
    X_train = convert_to_tensor(X_train, dtype=tf.float32)
    X_train = expand_dims(X_train, axis=-1)
    X_train = grayscale_to_rgb(X_train)
    X_train = resize(X_train, size=(128, 128))

    X_val = X_val.reshape(-1, 32, 32)
    X_val = convert_to_tensor(X_val, dtype=tf.float32)
    X_val = expand_dims(X_val, axis=-1)
    X_val = grayscale_to_rgb(X_val)
    X_val = resize(X_val, size=(128, 128))
    print(X_train.shape, X_val.shape)
    
    return X_train, y_train, X_val, y_val
    
    