IMDB Movie Reviews — LSTM Sentiment Classifier
End-to-end notebook that builds a binary sentiment classifier on the IMDB dataset using Keras/TensorFlow. It covers data loading, preprocessing (tokenization + padding), model training with an Embedding + LSTM stack, evaluation on the test set, and example predictions.

Features
Uses keras.datasets.imdb (50k reviews; labels: 0=negative, 1=positive)

Parameterized preprocessing: vocabulary size and max sequence length

LSTM-based model with Embedding → LSTM → Dense(sigmoid)

Training history visualization and test metrics

Decoding utility to view raw review text

Example predictions on held-out reviews

Project Structure
notebooks/ imdb_lstm_sentiment.ipynb — main notebook

README.md — project documentation (this file)

requirements.txt — minimal dependencies

Environment
Python 3.9+

TensorFlow 2.12+ (or compatible)

Keras (bundled with TF 2.x), NumPy, Matplotlib, scikit-learn (for metrics)

Install:

pip install -r requirements.txt

or: pip install tensorflow numpy matplotlib scikit-learn

Optional: Run on GPU for faster training.

Data
Loaded via: keras.datasets.imdb.load_data(num_words=vocab_size)

Word indices provided by Keras; utility maps indices back to tokens for inspection

Train/test split is provided (25k/25k)

Preprocessing
Limit vocabulary to vocab_size (default: 20,000)

Pad/truncate sequences to maxlen (default: 250)

pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

Optionally set random seeds for reproducibility

Model
Default architecture in the notebook:

Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen)

LSTM(128, dropout=0.2, recurrent_dropout=0.2)

Dense(1, activation='sigmoid')

Compile:

loss='binary_crossentropy'

optimizer='adam'

metrics=['accuracy']

Training:

batch_size=64

epochs=8

validation_split=0.2

callbacks: EarlyStopping(patience=2, restore_best_weights=True)

Results
The notebook prints:

Training/validation accuracy and loss per epoch

Final test accuracy

A few decoded reviews with predicted probabilities and labels

Typical baseline (may vary by run/hyperparameters):

Train acc: ~88–92%

Test acc: ~85–90%

Example Usage (notebook)
Open notebooks/imdb_lstm_sentiment.ipynb (Colab or local Jupyter)

Run all cells to:

Load and preprocess data

Build and train model

Evaluate on test set

Generate predictions for sample reviews

Customization
Use Bidirectional LSTM:

Bidirectional(LSTM(128, ...))

Tune hyperparameters:

vocab_size, maxlen, embedding_dim, lstm_units, batch_size, learning rate

Regularization:

Increase dropout, add L2 on Dense

Checkpointing:

ModelCheckpoint to save best weights

Visualization:

Add confusion matrix and classification report (scikit-learn)

Reproducibility
Set seeds in NumPy/TensorFlow; results can still vary due to nondeterministic GPU ops.

Limitations
Keras IMDB dataset is indexed, not raw text; pretrained embeddings require custom tokenization

LSTM can be slower than CNN/Transformer baselines; performance depends on hyperparameters and hardware


Acknowledgments
Keras/TensorFlow team for the IMDB dataset loader and deep learning APIs.
