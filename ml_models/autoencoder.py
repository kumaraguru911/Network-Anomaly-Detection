from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
import os

class AutoEncoder:
    def __init__(self, data_shape):
        self.data_shape = data_shape
    
        ## input layer 
        self.input_layer = Input(shape=(self.data_shape,))

        ## encoding part
        encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(self.input_layer)
        encoded = Dense(50, activation='relu')(encoded)

        ## decoding part
        decoded = Dense(50, activation='tanh')(encoded)
        decoded = Dense(100, activation='tanh')(decoded)

        ## output layer
        self.output_layer = Dense(self.data_shape, activation='relu')(decoded)

    def compile_model(self):
        #Compile the model
        self.autoencoder = Model(self.input_layer, self.output_layer)
        self.autoencoder.compile(optimizer="adadelta", loss="mse")

        return self.autoencoder

    def getHiddenRepresentation(self, model):
        #Let's try to get latent learnt representation by autoencoder
        hidden_representation = Sequential()
        hidden_representation.add(model.layers[0])
        hidden_representation.add(model.layers[1])
        hidden_representation.add(model.layers[2])

        return hidden_representation

    def save_load_models(self, path, model=None, mode="save"):
        # Add .keras extension if not present
        if not path.endswith('.keras') and not path.endswith('.h5'):
            model_path = path + '.keras'
        else:
            model_path = path
            
        if mode=="save":
            model.save(model_path)
        else:
            # Try loading with .keras extension first, then .h5 if that fails
            if os.path.exists(model_path):
                return load_model(model_path)
            elif os.path.exists(path + '.h5'):
                return load_model(path + '.h5')
            else:
                # Final fallback
                return load_model(path)
