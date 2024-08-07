# Ajouter le répertoire parent pour les imports de module
import sys
sys.path.append('..')

from src.data.params import MODEL_PARAMS, SEED
from src.data.make_dataset import get_dataset

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers.legacy import Adam
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


FEATURES = MODEL_PARAMS["DEFAULT_FEATURE_NAMES"]
TARGET_NAME = MODEL_PARAMS["TARGET_NAME"]
    
def read_and_split_data(features= ['keyword', 'text'], target='target'):
    # This function will split the data into training, validation, and testing sets
    # and return the required variables
    
    # Assuming train, test are the preprocessed datasets
    # and FEATURES, TARGET_NAME are defined as per the context
    train, test = get_dataset(raw=False)


    # Splitting the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        train.loc[:, FEATURES],
        train[TARGET_NAME],
        test_size=MODEL_PARAMS["TEST_SIZE"],
        random_state=SEED
    )
    
    # Assuming test dataset is already split and ready for use
    x_test = test.loc[:, FEATURES]
    
    # Fill NaN values in features column with empty string
    x_train = x_train.fillna('')
    x_val = x_val.fillna('')
    x_test = x_test.fillna('')
    
    # Concatenating the features columns
    x_train = x_train.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    x_val = x_val.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    x_test = x_test.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    return x_train.values, y_train.values, x_val.values, y_val.values, x_test.values

def train_pretrained_model(model_name='bert-base-uncased', use_class_weights=True, lr=5e-7):
    # Load pre-trained model and tokenizer
    x_train, y_train, x_val, y_val, x_test = read_and_split_data()
    
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare data for training
    train_encodings = tokenizer(x_train.tolist(), truncation=True, padding=True, return_tensors='tf')
    val_encodings = tokenizer(x_val.tolist(), truncation=True, padding=True, return_tensors='tf')

    # Compute class weights if required
    if use_class_weights:
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weights = {0: class_weights[0], 1: class_weights[1]}
    else:
        class_weights = None
    
    model.compile(optimizer=Adam(learning_rate=lr), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=SparseCategoricalAccuracy('accuracy'))

    return model, train_encodings, val_encodings, class_weights

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title('Confusion Matrix')
    plt.show()
