import os
# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import random, math, sys, signal
from shutil import copyfile
from importlib import reload
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# for debugging
#import ptvsd
#ptvsd.enable_attach(log_dir= os.path.dirname(__file__))
#ptvsd.wait_for_attach(timeout=15)

from util import options, data_processing
from ml_models import autoencoder

def train(args):
    print(args.model_name)
    
    #Prepare the data for training
    data_processor = data_processing.processData(args.data_path)
    X_train_scaled, Y_train, X_val_scaled, Y_val, scaler = data_processor.prepareTrainingData()
    
    # Pre-train AutoEncoder with normal data
    model = autoencoder.AutoEncoder(X_train_scaled.shape[1])
    compiled_model = model.compile_model() 
    print("model compiled")
    
    # Use the scaled training data
    # Select only normal data for autoencoder pre-training
    x_norm = X_train_scaled[Y_train == 0]
    x_fraud = X_train_scaled[Y_train == 1] # Define x_fraud here
    
    
    #Let's pretrain the compiled autoencoder with normal data, we don't need to train with every data point!
    # Use available normal samples (handle case when there are fewer samples)
    num_normal_samples = len(x_norm)
    print(f"Number of normal samples available: {num_normal_samples}")
    
    # Make sure we have enough samples for validation split (min 5 samples in validation)
    if num_normal_samples > 0:
        # Adjust batch size based on number of samples
        batch_size = min(256, max(32, num_normal_samples // 10))  # Reasonable batch size
        
        compiled_model.fit(x_norm, x_norm, 
                    batch_size=batch_size, epochs=50, 
                    shuffle=True, validation_split=0.20);
    else:
        print("Error: No normal samples found in the dataset. Check your data labels.")
        sys.exit(1)
    
    save_path = os.path.join(args.ckpt_path, args.model_name)
    model.save_load_models(path=save_path, model=compiled_model)
    del(compiled_model)
    compiled_model = model.save_load_models(path=save_path, mode="load")
    
    # Save the scaler as well
    scaler_filename = os.path.join(save_path, "scaler.pkl")
    pickle.dump(scaler, open(scaler_filename, 'wb'))

    # Now Let's try to get latent representation of the trained model
    hidden_representation = model.getHiddenRepresentation(compiled_model)

    # Dynamically use available samples for latent representation
    print(f"Normal samples: {len(x_norm)}, Fraud samples: {len(x_fraud)}")
    
    # Use all available samples or up to a reasonable limit
    max_samples = 6000  # Maximum samples per class to use
    
    # For normal samples
    if len(x_norm) > 0:
        # Use a percentage of available data instead of fixed indices
        norm_sample_count = min(len(x_norm), max_samples)
        norm_hid_rep = hidden_representation.predict(x_norm[:norm_sample_count])
        y_n = np.zeros(norm_hid_rep.shape[0])
    else:
        print("Warning: No normal samples available for latent representation")
        norm_hid_rep = np.array([]).reshape(0, hidden_representation.layers[-1].output_shape[1])
        y_n = np.array([])
    
    # For fraud samples
    if len(x_fraud) > 0:
        fraud_sample_count = min(len(x_fraud), max_samples)
        fraud_hid_rep = hidden_representation.predict(x_fraud[:fraud_sample_count])
        y_f = np.ones(fraud_hid_rep.shape[0])
    else:
        print("Warning: No fraud samples available for latent representation")
        fraud_hid_rep = np.array([]).reshape(0, hidden_representation.layers[-1].output_shape[1])
        y_f = np.array([])
    
    # Combine normal and fraud representations
    rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis=0) if norm_hid_rep.size > 0 and fraud_hid_rep.size > 0 else np.array([])
    rep_y = np.append(y_n, y_f)

    #Finally we can train a classifier on learnt representations

    train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
    clf = LogisticRegression(solver="liblinear", random_state=42).fit(train_x, train_y) # Use liblinear for smaller datasets, add random_state
    pred_y = clf.predict(val_x) 
    #Let's also save this classifier for future
    filename = os.path.join(save_path, "linear_regression_Classifier.pkl")
    s = pickle.dump(clf, open(filename, 'wb'))


    print("model is now pickled and saved for later")
    

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    print ("")
    print ("Classification Report: ")
    print (classification_report(val_y, pred_y))

    print ("")
    print ("Accuracy Score: ", accuracy_score(val_y, pred_y))



def test(args):
    #Step1 Read the test data, you can use the helper funtions from util that were also used duting training
    data_processor = data_processing.processData(args.data_path)
    
    # Load the scaler used during training
    save_path = os.path.join(args.ckpt_path, args.model_name)
    scaler_filename = os.path.join(save_path, "scaler.pkl")
    if not os.path.exists(scaler_filename):
        raise FileNotFoundError(f"Scaler not found at {scaler_filename}. Please train the model first.")
    scaler = pickle.load(open(scaler_filename, 'rb'))

    # Step2: Prepare and scale the test data using the loaded scaler
    X_test_scaled = data_processor.prepareTestData(scaler)
    
    # Initialize AutoEncoder model with the correct input shape
    model = autoencoder.AutoEncoder(X_test_scaled.shape[1])

    #Step3: Let's load the trained autoencoder into memory
    # Pre-train AutoEncoder with normal data
    compiled_model = model.compile_model() 
    compiled_model = model.save_load_models(path=save_path, mode="load")
    
    #Step4: Get latent representation of test data
    hidden_representation = model.getHiddenRepresentation(compiled_model)
    
    # Step 5 Load the Classifier now
    # load the model from disk
    filename = os.path.join(save_path, "linear_regression_Classifier.pkl")
    loaded_model = pickle.load(open(filename, 'rb'))
    #Step6: Get predictions and save to the data
    batch_size = 5000 # increase this according to the memory space you have
    num_samples = len(X_test_scaled)
    print(num_samples)
    predictions = []
    for idx in range(0, num_samples, batch_size):
        norm_hid_rep = hidden_representation.predict(X_test_scaled[idx: idx+batch_size])
        pred_y = loaded_model.predict(norm_hid_rep)
        predictions.extend(pred_y)
        
    data_processor.save_df(predictions, num_samples, "test_output.csv")

if __name__ == "__main__":
    args = options.ArgumentParser()

    if args.mode == "train":
            print("Training")
            # train()
            train(args)
    elif args.mode=="test":
            print("Testing")
            test(args)
    
    elif args.mode=="colab":
            pass
    else:
            raise Exception 