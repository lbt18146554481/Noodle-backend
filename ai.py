import base64
import cv2
import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from brokenaxes import brokenaxes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.layers import Lambda, Dense, Flatten, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Nadam, Ftrl
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class Dataset:
    """
    Dataset class for loading, saving, and processing image datasets.
    """
    
    def __init__(self, images=None, labels=None):
        self.images = images
        self.labels = labels
        self.filename = None

    def save_dataset(self, file_path, images=None, labels=None):
        if images is None:
            images = self.images
        if labels is None:
            labels = self.labels

        # Avoid subscript
        base_name = file_path.split('.')[0]
        if not file_path.endswith('.h5'):
            self.filename = f"{base_name}.h5"
        else:
            self.filename = file_path

        if (images is None) or (labels is None):
            raise ValueError("Images or Labels not loaded successfully.")
        else:
            with h5py.File(self.filename, 'w') as h5f:
                # Save images and labels directly
                h5f.create_dataset('images', data=images, dtype='float32')
                h5f.create_dataset('labels', data=labels.astype('S'))
        
        return self.filename

    def _load_csv(self, filename):
        self.dataset = pd.read_csv(filename)
        self.images = (self.dataset.iloc[:,1:].values).astype('float32') # all pixel values
        self.labels = self.dataset.iloc[:,0].values.astype('int32') # only labels i.e targets digits
        return self.images, self.labels

    def load_saved_dataset(self, filename):
        if filename.endswith('.csv'):
            return self._load_csv(filename)

        # Avoid subscript
        base_name = filename.split('.')[0]
        if not filename.endswith('.h5'):
            filename = f"{base_name}.h5"
        
        # Open the HDF5 file
        with h5py.File(filename, 'r') as h5f:
            # Load images and labels
            self.images = h5f['images'][:]
            self.labels = h5f['labels'][:]
        
        for i in range(len(self.labels)):
            self.labels[i] = self.labels[i].decode('utf-8')
        
        return self.images, self.labels

    def print_shapes(self):
        if (self.images is None) or (self.labels is None):
            raise ValueError("Images or Labels not loaded successfully.")
        else:
            # Print the shapes of images and labels
            print("Images Shape:", self.images.shape)
            print("Labels Shape:", self.labels.shape)
            return self.images.shape, self.labels.shape

    def find_random_image_per_class(self):
        if (self.images is None) or (self.labels is None):
            raise ValueError("Images or Labels not loaded successfully.")
        else:
            # Print the first image for each unique label
            label_list = []
            image_list = []
            
            df = pd.DataFrame({'image': list(self.images), 'label': self.labels})

        # Get unique labels
        unique_labels = df['label'].unique()

        # Initialize lists to hold the new images and labels
        image_list = []
        label_list = []

        # For each unique label, take a random image
        for label in unique_labels:
            # Filter the DataFrame for the current label
            label_images = df[df['label'] == label]
            # Randomly select one image from the filtered DataFrame
            random_image = label_images.sample(n=1)
            image_list.append(random_image['image'])
            label_list.append(random_image['label'].values[0])

        return image_list, label_list
    
    def encode_b64(self):
        b64_images = []

        for image in self.images:
            image_array = np.resize(image,(28,28)).astype('uint8')
            image = Image.fromarray(image_array)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            b64_images.append(img_str)

        self.images = np.array(b64_images)

        return self.images

    def decode_b64(self):
        images = []

        for b64_image in self.images:
            img_data = base64.b64decode(b64_image)
            buffered = BytesIO(img_data)
            image = Image.open(buffered)
            image = np.array(image)
            images.append(image)

        self.images = images

        return images
    
class Preprocessing:
    """
    Preprocessing class for image datasets.
    """

    def __init__(self, filename=None, X=None, y=None):
        self.X = X
        self.y = y
        # filename = file name of the dataset from current directory
        if filename is not None:
            dataset = Dataset()
            self.X, self.y = dataset.load_saved_dataset(filename)
        elif X is None:
            raise ValueError("No filename input or image input found.")

    def get_x(self):
        if self.X is not None:
            return self.X
        else:
            raise ValueError("No images found.")

    def get_y(self):
        if self.y is not None:
            return self.y
        else:
            raise ValueError("No labels found.")

    # Noise removal/ Outlier removal
    def noise_removal(self, alpha):
        raise NotImplementedError

    # Image Filtering/ Cropping
    def crop(self, size):
        raise NotImplementedError

    # Image Resizing (allow Multi-Resolution Training)
    def resize(self, width, height):
        # Specify the new size as (width, height)
        new_size = (width, height)
        old_X = self.X
        self.X = np.zeros((len(old_X), width, height, *old_X[0].shape[2:]), dtype=np.uint8)
        for i in range(len(old_X)):
            self.X[i] = cv2.resize(old_X[i], new_size)
        return self.X

    # Grayscale Conversion
    def convert_to_grayscale(self):
        # Use the cvtColor() function to grayscale the image
        width, height = self.X[0].shape[:2]
        if self.X[0].ndim == 3:
            if self.X[0].shape[2] == 1:
                # If it is already grayscale but with a 4-th dimension
                self.X = np.squeeze(self.X, axis=-1)
            else:
                # If it is RGB or RGBA
                old_X = self.X
                self.X = np.zeros((len(self.X), width, height), dtype=np.uint8)
                cv2.imwrite('ori.jpg', old_X[0])
                for i in range(len(old_X)):
                    self.X[i] = cv2.cvtColor(old_X[i], cv2.COLOR_BGR2GRAY)
        return self.X

    # Data Shuffling
    def shuffle_data(self):
        from sklearn.utils import shuffle
        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        return self.X, self.y

    # Normalization
    def normalize(self):
        from sklearn.preprocessing import MinMaxScaler
        # Initialize the scaler
        scaler = MinMaxScaler()   
        # Fit and transform the data
        for i in range(len(self.X)):
            self.X[i] = scaler.fit_transform(self.X[i])
        return self.X

    def save_dataset(self, filename):
        dataset = Dataset(self.X, self.y)
        dataset.save_dataset(filename)

    def return_class_example(self):
        dataset = Dataset(self.X, self.y)
        images, labels = dataset.find_random_image_per_class()
        images = dataset.encode_b64()
        class_example = {label: image for label, image in zip(labels, images.tolist())}
        return class_example

class CNN:
    """
    Convolutional Neural Network class for training and evaluating models.
    """

    def __init__(self, X=None, y=None, training_options={"method": "train_test_val","layers": [{"type": "Flatten"},  {"type": "Dense", "units": 512, "activation": "relu"}, {"type": "Dense", "units": 10, "activation": "softmax"}],"optimizer": "Adam","loss": "categorical_crossentropy","metrics": ["accuracy"],"lr": 0.01,"epochs": 10,"batch_size": 64,"kFold_k": 5}, model = None):
        self.X = X
        self.y = y
        self.model = model
        self.hist = None
        self.batches = None
        self.val_batches = None
        self.layers = training_options['layers']
        self.optimizer = training_options['optimizer']
        self.loss = training_options['loss']
        self.metrics = ['accuracy']
        self.lr = training_options['lr']
        self.epochs = training_options['epochs']
        self.batch_size = training_options['batch_size']
        self.method = training_options['method']
        self.kFold_k = training_options['kFold_k']

    def train_model(self):
        self.y = to_categorical(self.y)
        if self.method=="train_test":
            self._train_test_approach()
        elif self.method=="train_test_val":
            self._train_test_val_approach()
        elif self.method=="kFold_val":
            self._kFold_validation_approach()
        else:
            raise ValueError(f"Invalid method '{self.method}'. Expected one of: 'train_test', 'train_test_val', 'kFold_val'.")

        return self.model

    # train/test approach
    def _train_test_approach(self):
        # train/test split
        seed = 43
        np.random.seed(seed)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Make sure grayscale images have 4-th dimension
        if self.X_train.ndim == 3:
            self.X_train = np.expand_dims(self.X_train, axis=-1)
        if self.X_test.ndim == 3:
            self.X_test = np.expand_dims(self.X_test, axis=-1)

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

        # Make sure grayscale images have 4-th dimension
        if self.X_train.ndim == 3:
            self.X_train = np.expand_dims(self.X_train, axis=-1)
        if self.X_val.ndim == 3:
            self.X_val = np.expand_dims(self.X_val, axis=-1)
        if self.X_test.ndim == 3:
            self.X_test = np.expand_dims(self.X_test, axis=-1)

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
        self.model = Sequential()
        for layer in self.layers:
            self._add_layer(layer)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.model.optimizer.lr=self.lr
        # self.model.optimizer.clipnorm=1
        # FIXME: self.model.optimizer.clipvalue=0.5

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

        # Save the loss and validation loss
        self.loss_data = pd.DataFrame({
            'Epoch': epochs,
            'Loss': loss_values,
            'Val_Loss': val_loss_values
        })

        # Save the accuracy and validation accuracy
        self.accuracy_data = pd.DataFrame({
            'Epoch': epochs,
            'Accuracy': acc_values,
            'Val_Accuracy': val_acc_values
        })

        # Create a figure for loss
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, loss_values, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b+', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.ylim(0, 1)

        # Save the loss figure
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        # Convert the BytesIO object to a NumPy array
        buf.seek(0)  # Move to the beginning of the BytesIO object
        image = Image.open(buf)
        self.loss_graph = image

        # Create a figure for accuracy
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_acc_values, 'b+', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.ylim(0, 1)

        # Save the accuracy figure
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        # Convert the BytesIO object to a NumPy array
        buf.seek(0)  # Move to the beginning of the BytesIO object
        image = Image.open(buf)
        self.accuracy_graph = image

    def _plot_kfold_performance(self, avg_loss, avg_val_loss, avg_acc, avg_val_acc):
        epochs = range(1, len(avg_loss) + 1)

        # Save the loss and validation loss
        self.loss_data = pd.DataFrame({
            'Epoch': epochs,
            'Loss': avg_loss,
            'Val_Loss': avg_val_loss
        })

        # Save the accuracy and validation accuracy
        self.accuracy_data = pd.DataFrame({
            'Epoch': epochs,
            'Accuracy': avg_acc,
            'Val_Accuracy': avg_val_acc
        })

        # Create a figure for loss
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, avg_loss, 'bo', label='Training Loss')
        plt.plot(epochs, avg_val_loss, 'b+', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.ylim(0, 1)

        # Save the loss figure
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        # Convert the BytesIO object to a NumPy array
        buf.seek(0)  # Move to the beginning of the BytesIO object
        image = Image.open(buf)
        self.loss_graph = image

        # Create a figure for accuracy
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, avg_acc, 'bo', label='Training Accuracy')
        plt.plot(epochs, avg_val_acc, 'b+', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.ylim(0, 1)

        # Save the accuracy figure
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        # Convert the BytesIO object to a NumPy array
        buf.seek(0)  # Move to the beginning of the BytesIO object
        image = Image.open(buf)
        self.accuracy_graph = image

    def save_model(self, filename):
        self.model.save(filename)
        
    def load_model(self, filename):
        self.model = load_model(filename)

    def run_model(self, image):
        # Run model to predict the result of one image
        predictions = self.model.predict(image, verbose=1)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)

        return predicted_class, confidence

    def test_model(self, X_test, y_test):
        # Predict on testing data
        y_predict, _ = self.run_model(X_test)
        accuracy = accuracy_score(y_test, y_predict)

        # Find accuracy per class
        unique_labels = set(y_test)
        data_per_class =  {label: 0 for label in unique_labels}
        true_data_per_class = {label: 0 for label in unique_labels}

        for i in range(len(y_test)):
            data_per_class[y_test[i]] += 1
            if y_test[i] == y_predict[i]:
                true_data_per_class[y_test[i]] += 1

        accuracy_per_class = {key.item(): true_data_per_class[key] / data_per_class[key] for key in data_per_class}
        
        return accuracy, accuracy_per_class

    def get_performance_graphs(self):
        buffered = BytesIO()
        self.accuracy_graph.save(buffered, format="PNG")
        self.accuracy_graph = base64.b64encode(buffered.getvalue()).decode('utf-8')
        self.loss_graph.save(buffered, format="PNG")
        self.loss_graph = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return self.loss_graph, self.accuracy_graph

    def get_performance_data(self):
        return self.loss_data, self.accuracy_data
