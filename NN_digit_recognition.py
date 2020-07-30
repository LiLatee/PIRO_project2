from matplotlib import pyplot as plt
import keras
from keras.utils import to_categorical
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns


class CNN_model:
    

    
    def __init__(self):
        self.epochs = 20
        self.batch_size = 64
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same", input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), activation="relu", padding="same"))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), activation="relu", padding="same"))
        
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation="relu"))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation="softmax"))
        
    def learning(self,X_train, X_valid, y_train, y_valid,model_name):
        learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        data_aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
        history = self.model.fit_generator(data_aug.flow(X_train, y_train, batch_size=self.batch_size), 
                             epochs=self.epochs,
                             validation_data=(X_valid, y_valid),
                             steps_per_epoch=len(X_train) // self.batch_size,
                             callbacks=[learning_rate_reduction])
        self.model.save('models/{0}'.format(modelName))

        plt.subplots(figsize=(10, 12))

        plt.subplot(211)
        plt.title("Loss")
        loss = history.history["loss"]
        plt.plot(range(1, len(loss) + 1), loss, "bo-", label="Training Loss")
        loss = history.history["val_loss"]
        plt.plot(range(1, len(loss) + 1), loss, "ro-", label="Validation Loss")
        plt.xticks(range(1, len(loss) + 1))
        plt.grid(True)
        plt.legend()

        plt.subplot(212)
        plt.title("Accuracy")
        acc = history.history["acc"]
        plt.plot(range(1, len(loss) + 1), acc, "bo-", label="Training Acc")
        acc = history.history["val_acc"]
        plt.plot(range(1, len(loss) + 1), acc, "ro-", label="Validation Acc")
        plt.xticks(range(1, len(loss) + 1))
        plt.grid(True)
        plt.legend()
        
    def load_model(self, model_name = 'model_4'):
        self.model = keras.models.load_model('models/'+model_name)
        print("SUCC LOADED")
    
    def predict(self,X_valid,y_valid,show=True):
        pred = self.model.predict(X_valid)
        pred_classes = np.argmax(pred, axis=1)
#         print(pred)
        pred_true = np.argmax(y_valid, axis=1)
        if show:
            confusion_mtx = confusion_matrix(pred_true, pred_classes)
            print(confusion_mtx)

            sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap=plt.cm.Blues)
            cm = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]
            print(cm.diagonal())
        return pred_classes

    def predict_only_X(self,X_valid):
        pred = self.model.predict(X_valid)
        pred_classes = np.argmax(pred, axis=1)
        return pred_classes


    def save_model(self,model_name = 'model_100'):
        self.model.save('models/'+model_name)
        print("SUCC SAVED")

# def teach_new_model(modelName = 'model_1'):
    
#     seed = 1
#     np.random.seed(SEED)
#     sns.set(style="white", context="notebook", palette="deep")
    
#     train = pd.read_csv("../data/train.csv")
#     test = pd.read_csv("../data/test.csv")
    
#     y_train = train["label"]
#     X_train = train.drop(labels=["label"], axis=1)
#     X_train /= 255
#     test /= 255

#     X_train = X_train.to_numpy().reshape(-1, 28, 28, 1)
#     test = test.to_numpy().reshape(-1, 28, 28, 1)
#     y_train = to_categorical(y_train)

#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=SEED, train_size=0.9)
    
#     cnn_mod = CNN_model()
#     cnn_mod.learning(X_train, X_valid, y_train, y_valid,model_name)
    