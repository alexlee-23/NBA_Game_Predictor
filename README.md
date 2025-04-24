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
