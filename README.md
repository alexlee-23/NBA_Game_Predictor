## NBA Game Outcome Prediction ##
### CS6140 Final Project – by Alex Lee ###

### Overview ###
This project explores the use of recent game histories to predict the outcomes of NBA games. I focus on modeling the past five games for both the home and opposing teams and use these temporal patterns to classify whether the home team will win.
The prediction task is framed as a binary classification problem, and I evaluate both classical and deep learning models, including:

- Logistic Regression
- Gaussian Naive Bayes
- Feedforward Neural Networks (FFNs)
- Long Short-Term Memory (LSTM) networks

My hypothesis is that incorporating temporal context and using dual-path architectures—which separately process each team's historical performance—can improve prediction accuracy compared to traditional models that treat games independently.

### Code ###
- All of my classifier training code primarily resides in the **./temporal_models_training** directory.
- In the **./data_sandboxing** directory resides code I wrote to learn the structure of the data before doing feature engineering on it. 
- In the **./DNN_Architectures** directory resides the different NN architectures that I built for temporal modeling

## How to Run Temporal Models

> **Important**: Make sure you have Git LFS installed, as the dataset file `NBA_temporal_dataset.pkl` is large and tracked using LFS.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Navigate to the Training Notebooks

Despite the many directories, the main notebooks for model training and generating individual analysis plots are all in:

```
./temporal_models_training
```

### 3. Run the Notebooks

Open each notebook and run all cells to train a model and generate corresponding plots.

For example, to run the **Dual-path LSTM**, open the file:

```
dual_lstm_temporal.ipynb
```

Then click **"Run All"** in Jupyter Notebook (or your preferred interface) and wait until training and evaluation are complete.

All trained models are saved into the directory from which you run the notebook:

```
./models
```

### 4. Notebook Reference Table

| Model                            | Notebook File                  |
|----------------------------------|--------------------------------|
| Dual-path Feed Forward Network   | `dual_ffn_temporal.ipynb`     |
| Single Feed Forward Network      | `fffn_temporal.ipynb`         |
| Dual-path LSTM Network           | `dual_lstm_temporal.ipynb`    |
| Single LSTM Network              | `lstm_temporal.ipynb`         |
| Logistic Regression & GNB        | `naive_temporal.ipynb`        |


## Requirements

Make sure you have the following Python libraries installed:

```bash
numpy
pickle
pandas
scikit-learn
matplotlib
seaborn
torch
torchvision
tqdm
jupyter
```
