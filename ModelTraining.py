import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Lambda, Dense, Flatten, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Nadam, Ftrl
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class LoadProcessedData:
    def __init__(self, X=None, y=None):
        self.dataset = None
        self.X = None
        self.y = None

    def load_csv(self, filename):
        self.dataset = pd.read_csv(filename)
        self.X = (self.dataset.iloc[:,1:].values).astype('float32') # all pixel values
        self.y = self.dataset.iloc[:,0].values.astype('int32') # only labels i.e targets digits
        self.X = self.X.reshape(self.X.shape[0], 28, 28)
        self.X = self.X.reshape(self.X.shape[0], 28, 28,1)
        return self.X, self.y

    def encode_label(self):
        self.y = to_categorical(self.y)
        return self.X, self.y

class CNN:
    def __init__(self, X=None, y=None, method="train_test_val", layers=[{"type": "Flatten"},  {"type": "Dense", "units": 512, "activation": "relu"}, {"type": "Dense", "units": 10, "activation": "sofftmax"}], optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"], lr=0.01, epochs=10, batch_size=64, kFold_k=5):
        self.X = X
        self.y = y
        self.model = None
        self.hist = None
        self.batches = None
        self.val_batches = None
        self.layers=layers
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.lr=lr
        self.epochs=epochs
        self.batch_size=batch_size
        self.method=method
        self.kFold_k=kFold_k

    def train_model(self):
        if self.method=="train_test":
            self._train_test_approach()
        elif self.method=="train_test_val":
            self._train_test_val_approach()
        elif self.method=="kFold_val":
            self._kFold_validation_approach()
        else:
            raise ValueError(f"Invalid method '{self.method}'. Expected one of: 'train_test', 'train_test_val', 'kFold_val'.")

    def _standardize(self, x):
        mean_px = self.X.mean().astype(np.float32)
        std_px = self.X.std().astype(np.float32)
        return (x-mean_px)/std_px
    
    # train/test approach
    def _train_test_approach(self):
        # train/test split
        seed = 43
        np.random.seed(seed)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Create Data Generator
        gen = ImageDataGenerator()
        self.batches = gen.flow(self.X_train, self.y_train, batch_size=self.batch_size)
        self.val_batches = gen.flow(self.X_test, self.y_test, batch_size=self.batch_size)

        # Define and train model
        self._custom_model()
        
        # Plot performance of the model
        self._check_performance()

    # train/test/val
    def _train_test_val_approach(self):
        # train/val/test split
        seed = 43
        np.random.seed(seed)
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Create Data Generator
        gen = ImageDataGenerator()
        self.batches = gen.flow(self.X_train, self.y_train, batch_size=self.batch_size)
        self.val_batches=gen.flow(self.X_val, self.y_val, batch_size=self.batch_size)

        # Define and train model
        self._custom_model()
        
        # Plot performance of the model
        self._check_performance()
        test_generator = gen.flow(self.X_test, self.y_test, batch_size=self.batch_size)
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # k-fold CV
    def _kFold_validation_approach(self):
        # not yet finished
        kf = KFold(n_splits=self.kFOld_k, shuffle=True, random_state=42)
        
        # Create Data Generator
        gen = ImageDataGenerator()
        model_histories = []
        
        for train_index, val_index in kf.split(self.X):
            self.X_train, self.X_val = self.X[train_index], self.X[val_index]
            self.y_train, self.y_val = self.y[train_index], self.y[val_index]
            
            self.batches = gen.flow(self.X_train, self.y_train, batch_size=self.batch_size)
            self.val_batches=gen.flow(self.X_val, self.y_val, batch_size=self.batch_size)

            # Define and train model
            self._custom_model()
            
            # Plot performance of the model
            avg_loss = np.mean([history.history['loss'] for history in model_histories], axis=0)
            avg_val_loss = np.mean([history.history['val_loss'] for history in model_histories], axis=0)
            avg_accuracy = np.mean([history.history['accuracy'] for history in model_histories], axis=0)
            avg_val_accuracy = np.mean([history.history['val_accuracy'] for history in model_histories], axis=0)
            self._plot_kfold_performance(avg_loss, avg_val_loss, avg_accuracy, avg_val_accuracy)

    def _custom_model(self):
        self.model= Sequential([Lambda(self._standardize, input_shape=self.X[0].shape)])
        for layer in self.layers:
            self._add_layer(layer)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            
        self.model.optimizer.lr=self.lr
        
        self.hist = self.model.fit(
            x=self.batches,
            steps_per_epoch=self.batches.n,
            epochs=self.epochs,
            validation_data=self.val_batches,
            validation_steps=self.val_batches.n,
            verbose=1
        )

        return self.model

    def _add_layer(self, layer):
        if layer["type"] == "Flatten":
            self.model.add(Flatten())
        elif layer["type"] == "Dense":
            if "activation" in layer and layer["activation"] is not None:
                self.model.add(Dense(layer["units"], activation=layer["activation"]))
            else:
                self.model.add(Dense(layer["units"]))
        elif layer["type"] == "Activation":
            self.model.add(Activation(layer["activation"]))
        elif layer["type"] == "Dropout":
            self.model.add(Dropout(layer["rate"]))
        elif layer["type"] == "BatchNormalization":
            self.model.add(BatchNormalization())
        elif layer["type"] == "Conv2D":
            self.model.add(Conv2D(layer["number of filters"], layer["kernel_size"], activation=layer.get("activation"), padding=layer.get("padding", "valid")))
        elif layer["type"] == "MaxPooling2D":
            self.model.add(MaxPooling2D(pool_size=layer["pool_size"]))
        elif layer["type"] == "GlobalMaxPooling2D":
            self.model.add(GlobalMaxPooling2D())
        elif layer["type"] == "AveragePooling2D":
            self.model.add(AveragePooling2D(pool_size=layer["pool_size"]))
        elif layer["type"] == "GlobalAveragePooling2D":
            self.model.add(GlobalAveragePooling2D())

    def _check_performance(self):
        history_dict = self.hist.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        epochs = range(1, len(loss_values) + 1)

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot loss
        axs[0].plot(epochs, loss_values, 'bo', label='Training Loss')
        axs[0].plot(epochs, val_loss_values, 'b+', label='Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss Over Epochs')
        axs[0].legend()
        axs[0].set_ylim(0, 1)

        # Plot accuracy
        axs[1].plot(epochs, acc_values, 'bo', label='Training Accuracy')
        axs[1].plot(epochs, val_acc_values, 'b+', label='Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy Over Epochs')
        axs[1].legend()
        axs[1].set_ylim(0, 1)

        # Save the figure
        plt.tight_layout()
        plt.savefig('performance_plot.png')
        plt.close()
    
    def _plot_kfold_performance(self, avg_loss, avg_val_loss, avg_acc, avg_val_acc):
        epochs = range(1, len(avg_loss) + 1)

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot loss
        axs[0].plot(epochs, avg_loss, 'bo', label='Average Training Loss')
        axs[0].plot(epochs, avg_val_loss, 'b+', label='Average Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Average Loss Over Epochs')
        axs[0].legend()

        # Plot accuracy
        axs[1].plot(epochs, avg_acc, 'bo', label='Average Training Accuracy')
        axs[1].plot(epochs, avg_val_acc, 'b+', label='Average Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Average Accuracy')
        axs[1].set_title('Average Accuracy Over Epochs')
        axs[1].legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig('performance_plot.png')
        plt.close()
      
    def run_model(self, X_test):
        X_test = X_test.reshape(len(X_test), 28, 28, 1).astype('float32')
        predictions = self.model.predict(X_test, verbose=1)
        predicted_class = np.argmax(predictions, axis=1)
        print(predicted_class)
        return predicted_class
