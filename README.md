# Network-Anomaly-Detection

This project aims to provide a setup for anomaly detection in Networking, specifically to detect DDoS attacks using a Keras/TensorFlow-based Autoencoder.

## Introduction
Many IoT devices are becoming victims of hackers due to their lack of security and they are often turned into botnets conducting Distributed Denial of Service (DDoS) attacks. We aim to detect those attacks by analyzing their network traffic. 

When designing the model, one has to keep in mind that in a real life scenario, the attack detection is relevant only if it is conducted in a streaming/near real time way.

## Core Idea
We will create an autoencoder model in which we only show the model non-fraud cases. The model will try to learn the best representation of normal cases. The same model will be used to generate the representations of cases where a DDoS attack is done, and we expect them to be different from normal ones.

![AutoEncoder](extra/autoencoder-net-arch.png) 

The beauty of this approach is that the autoencoder learns the characteristics of normal traffic. When presented with anomalous traffic (like a DDoS attack), it will fail to reconstruct it accurately, resulting in a high reconstruction error. This error becomes the basis for our detection.

Once the autoencoder is trained, we use its internal "latent representation" as features to train a simple and fast `LogisticRegression` classifier. This two-step process allows for effective and efficient anomaly detection.

## Results

The model achieves high accuracy in distinguishing between normal (Benign) and DDoS attack traffic. The classification report below shows the performance on a validation set after training.

!AutoEncoderResults 

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/kumaraguru911/Network-Anomaly-Detection.git
cd Network-Anomaly-Detection
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows (PowerShell/CMD):
.\venv\Scripts\Activate

# On macOS/Linux:
# source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

This project does not include a dataset. You will need to download one. The code is optimized for the **CIC-IDS-2017 dataset**.

1.  **Download**: Visit the CIC-IDS-2017 Dataset Page and download a CSV file (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`).
2.  **Organize**:
    *   Create a `data` folder in the project's root directory.
    *   Move the downloaded CSV file into this folder and rename it to `data.csv`.
    *   The final path should be `data/data.csv`.

### 5. Train the Model

Run the main script in `train` mode. This will process the data, train the autoencoder and classifier, and save the trained models to the `checkpoints/` directory.

```bash
python main.py --mode train --data_path data/data.csv
```

### 6. Test the Model

First, create a sample test file from your main dataset using the provided script.

```bash
python create_test_data.py
```

Now, run the main script in `test` mode. This will load the trained models, make predictions on `data/test_data.csv`, and save the results to `test_output.csv`.

```bash
python main.py --mode test --data_path data/test_data.csv
```

## Visualizing Data Relationship with TSNE:

T-SNE (t-Distributed Stochastic Neighbor Embedding) is a dataset decomposition technique which reduced the dimentions of data and produces only top n components with maximum information.

Every dot in the following represents a request. Normal transactions are represented as Green while potential attacks are represented as Red. The two axis are the components extracted by tsne.

The visualization on the right clearly shows how the autoencoder's latent representation creates a much better separation between the two classes, making it easier for a simple classifier to do its job.

| TSNE on Raw Scaled Data | TSNE on Autoencoder's Latent Representation |
| :---------------------: | :-----------------------------------------: |
| !TSNE | !TSNE-1 |

## Future Work

- Adding and training more models
- further classifying different kinds of DDoS attacks!
- Creating a semi-supervised loop to update models online while new data is being gathered.

