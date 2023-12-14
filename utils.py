
import config
import pandas as pd
import os
import shutil
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import convert_to_tensor, expand_dims
from tensorflow.image  import grayscale_to_rgb, resize
from matplotlib import pyplot as plt


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
  
def plot_train_accuracy(history, name):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0.2, 1])
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'])
    name = name + '.png'
    plt.savefig(name, dpi=600)
    
def plot_train_loss(history, name):
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'])
    plt.savefig('ResNet50_loss.png', dpi=600)
    name = name + '.png'
    plt.savefig(name, dpi=600)
 
 
def plot_train_history(history, name):
    name1 = name + '_accuracy'
    name2 = name + '_loss'
    plot_train_accuracy(history, name1) 
    plot_train_loss(history, name2)

def performance(model, model_name,  X_val, y_val):
    pred = model.predict(X_val)
    pred = np.argmax(pred, axis=1)
    report = classification_report(y_val, pred, output_dict=True)

    print("Report:\n", report)
    df_report = pd.DataFrame(report)
    df_report = df_report.round(4)
    name =model_name +  '.csv'
    df_report.to_csv(name, index=False)

    cm = confusion_matrix(y_val, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',  xticklabels=[x for x in range(28)], yticklabels=[x for x in range(28)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    name = model_name + ',ong'
    plt.savefig(name, dpi=600)
    
    print(cm)
    

def save_trained_model(model, name):
    model.save(name, save_format='tf')
    folder_path = config.trained_models_path
    os.makedirs(folder_path, exist_ok=True)
    folder_path =  folder_path + '/' + name
    shutil.make_archive(folder_path, 'zip', folder_path)