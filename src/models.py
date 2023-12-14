from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19, MobileNetV2, ResNet50
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, Flatten
from tensorflow import nn as nn
import config



class VGG19Backend(Model):
    def __init__(self, num_classes=config.num_classes):
        super(VGG19Backend, self).__init__()

        self.base_model = VGG19(
            weights='imagenet',
            input_shape=config.input_shape,
            include_top=False
        )
        for layer in self.base_model.layers:
            layer.trainable = False

        self.conv1 = Conv2D(64, (3, 3), padding='same')
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(128, (3, 3), padding='same')
        self.batch_norm2 = BatchNormalization()
        self.conv3 = Conv2D(256, (3, 3), padding='same')
        self.batch_norm3 = BatchNormalization()

        self.flatten = Flatten()

        # Additional Dense Layers
        self.dense1 = Dense(512)
        self.batch_norm_dense1 = BatchNormalization()
        self.dense2 = Dense(256)
        self.batch_norm_dense2 = BatchNormalization()
        self.dense3 = Dense(128)
        self.batch_norm_dense3 = BatchNormalization()
        self.dense4 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = nn.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm_dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = self.batch_norm_dense2(x)
        x = nn.relu(x)
        x = self.dense3(x)
        x = self.batch_norm_dense3(x)
        x = nn.relu(x)
        output = self.dense4(x)
        return output


class ResNet50Backend(Model):
    def __init__(self, num_classes=config.num_classes):
        super(ResNet50Backend, self).__init__()

        self.base_model = ResNet50(
            weights='imagenet',
            input_shape=config.input_shape,
            include_top=False
        )
        for layer in self.base_model.layers:
            layer.trainable = False

        self.conv1 = Conv2D(64, (3, 3), padding='same')
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(128, (3, 3), padding='same')
        self.batch_norm2 = BatchNormalization()
        self.conv3 = Conv2D(256, (3, 3), padding='same')
        self.batch_norm3 = BatchNormalization()

        self.flatten = Flatten()

        # Additional Dense Layers
        self.dense1 = Dense(512)
        self.batch_norm_dense1 = BatchNormalization()
        self.dense2 = Dense(256)
        self.batch_norm_dense2 = BatchNormalization()
        self.dense3 = Dense(128)
        self.batch_norm_dense3 = BatchNormalization()
        self.dense4 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = nn.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm_dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = self.batch_norm_dense2(x)
        x = nn.relu(x)
        x = self.dense3(x)
        x = self.batch_norm_dense3(x)
        x = nn.relu(x)
        output = self.dense4(x)
        return output


class MobileNetV2Backend(Model):
    def __init__(self, num_classes=config.num_classes):
        super(MobileNetV2Backend, self).__init__()

        self.base_model = MobileNetV2(
            weights='imagenet',
            input_shape=config.input_shape,
            include_top=False
        )
        for layer in self.base_model.layers:
            layer.trainable = False

        self.conv1 = Conv2D(64, (3, 3), padding='same')
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(128, (3, 3), padding='same')
        self.batch_norm2 = BatchNormalization()
        self.conv3 = Conv2D(256, (3, 3), padding='same')
        self.batch_norm3 = BatchNormalization()

        self.flatten = Flatten()

        # Additional Dense Layers
        self.dense1 = Dense(512)
        self.batch_norm_dense1 = BatchNormalization()
        self.dense2 = Dense(256)
        self.batch_norm_dense2 = BatchNormalization()
        self.dense3 = Dense(128)
        self.batch_norm_dense3 = BatchNormalization()
        self.dense4 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = nn.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm_dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = self.batch_norm_dense2(x)
        x = nn.relu(x)
        x = self.dense3(x)
        x = self.batch_norm_dense3(x)
        x = nn.relu(x)
        output = self.dense4(x)
        return output
