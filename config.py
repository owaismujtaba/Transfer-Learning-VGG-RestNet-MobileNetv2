
import os

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')
trained_models_path = os.path.join(cur_dir, 'Models')

X_train_path = os.path.join(data_dir, 'csvTrainImages 13440x1024.csv')
X_test_path = os.path.join(data_dir,'csvTestImages 3360x1024.csv')

y_train_path = os.path.join(data_dir,'csvTrainLabel 13440x1.csv')
y_test_path = os.path.join(data_dir,'csvTestLabel 3360x1.csv')

image_shape = (128, 128)
input_shape = (128, 128, 3)

batch_size = 128
epochs = 1

vgg16 = True
restnet = False
monilenet = False

num_classes = 28



import pdb

