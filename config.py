
X_train_path = r'/content/drive/MyDrive/Arabic Character Recognition/Data/csvTrainImages 13440x1024.csv'
X_test_path = r'/content/drive/MyDrive/Arabic Character Recognition/Data/csvTestImages 3360x1024.csv'

y_train_path = r'/content/drive/MyDrive/Arabic Character Recognition/Data/csvTrainLabel 13440x1.csv'
y_test_path = r'/content/drive/MyDrive/Arabic Character Recognition/Data/csvTestLabel 3360x1.csv'

batch_size = 128
epochs = 100

vgg16 = True
restnet = False
monilenet = False

num_classes = 28