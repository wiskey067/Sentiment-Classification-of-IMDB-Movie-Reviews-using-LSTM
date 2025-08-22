# 🎬 IMDB Movie Reviews — LSTM Sentiment Classifier

An end-to-end deep learning project that builds a **binary sentiment classifier** on the [IMDB dataset](https://keras.io/api/datasets/imdb/) using **Keras + TensorFlow**.  
The model classifies reviews as **positive (1)** or **negative (0)** using an **Embedding → LSTM → Dense(sigmoid)** architecture.  

---

## 📌 Features
- ✅ Uses `keras.datasets.imdb` (50k reviews; pre-tokenized, labeled data)  
- ✅ Preprocessing with **tokenization + padding** (`pad_sequences`)  
- ✅ **LSTM-based model** for sequential text classification  
- ✅ Training history visualization (accuracy & loss curves)  
- ✅ **Example predictions** on held-out test reviews  
- ✅ Utility to decode word indices back to raw text  

---

## 📂 Project Structure
notebooks/
└── imdb_lstm_sentiment.ipynb   # Main notebook
README.md                        # Project documentation
requirements.txt                 # Minimal dependencies
---

## ⚙️ Environment
- Python 3.9+  
- TensorFlow 2.12+ (or compatible)  
- NumPy, Matplotlib, scikit-learn  

📊 Dataset
	•	Loaded via:
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
Train/Test split: 25k / 25k reviews
•	Preprocessing:
	•	Vocabulary limited to vocab_size (default: 20,000)
	•	Pad/truncate sequences to maxlen (default: 250)
 from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=250, padding="post", truncating="post")
🧠 Model Architecture
Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen)
LSTM(128, dropout=0.2, recurrent_dropout=0.2)
Dense(1, activation='sigmoid')
Compilation 
loss='binary_crossentropy'
optimizer='adam'
metrics=['accuracy']
Training
	•	batch_size=64, epochs=8
	•	validation_split=0.2
	•	Early stopping with patience=2
 
 📈 Results

Typical performance (varies with hyperparameters & hardware):
	•	Training Accuracy: 88–92%
	•	Test Accuracy: 85–90%

The notebook prints:
	•	Training/validation accuracy and loss per epoch
	•	Final test accuracy
	•	Example decoded reviews with predicted sentiment

🚀 Example Usage

Run the notebook locally or on Google Colab: jupyter notebook notebooks/imdb_lstm_sentiment.ipynb
You will be able to:
	1.	Load and preprocess IMDB dataset
	2.	Train and evaluate the LSTM model
	3.	Visualize accuracy & loss curves
	4.	Run predictions on example reviews
 🎛 Customization
	•	Model: Try Bidirectional(LSTM(...)) for stronger performance
	•	Hyperparameters: Adjust vocab_size, maxlen, embedding_dim, lstm_units, batch_size
	•	Regularization: Increase dropout, add L2 penalty
	•	Visualization: Add confusion matrix & classification report with scikit-learn
	•	Checkpointing: Save best model weights with ModelCheckpoint
 ⚠️ Limitations
	•	The IMDB dataset is pre-indexed (not raw text). Using pretrained embeddings (GloVe, Word2Vec) requires custom tokenization.
	•	LSTMs are slower than CNNs or Transformers; performance depends on hardware and tuning.

⸻

🙏 Acknowledgments
	•	Keras/TensorFlow team for the IMDB dataset loader and deep learning APIs.
