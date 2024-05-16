import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras

TEXT_COLUMNS = [
    'CollisionT',
    'CrashType',
    'TrafficCon',
    'CountyName',
    'CityTownNa'
]
TARGET_COLUMNS = [
    'properties.delay',
    'properties.length'
]

# Tokenize and pad each column
max_length = 50
col_tokenizers = []
padded_sequences_list = []
vocab_sizes = []
df = pd.read_csv('encoded_model_data.csv')
df.reset_index(drop=True, inplace=True)
for col in TEXT_COLUMNS:
    print(f'EMBEDDING COLUMN {col}')
    print(f'ORIGINAL DATA:\n{df[col].iloc[:5]}')
    # Create a tokenizer instance
    tokenizer = Tokenizer(oov_token='<UNK>')
    # Fit the tokenizer to the current col and add it to the list
    tokenizer.fit_on_texts(df[col])
    col_tokenizers.append(tokenizer)
    # Get the sequence of integers with the custom fit tokenizer, pad it, and add it to the list
    sequences = tokenizer.texts_to_sequences(df[col])
    print(f'SEQUENCES:\n{sequences[:5]}')
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    print(f'PADDED SEQUENCES:\n{padded_sequences[:5]}')
    padded_sequences_list.append(padded_sequences)
    print(f'WORD INDEX:\n{tokenizer.word_index}')
    vocab_sizes.append(len(tokenizer.word_index) + 1)

# Embedding layers for each text column
embedding_dim = 100
# Creates embedding layers (one for each col that needs embedding)
# Input size has to be equal to size of vocab + 1
embedding_layers = [keras.layers.Embedding(length, embedding_dim, input_length=max_length) for length in vocab_sizes]

num_col = [col for col in df.columns if (col not in TEXT_COLUMNS) and ('Unnamed' not in col) and (col not in TARGET_COLUMNS)]
# Create an input layer with the number of nodes equal to the number of numerical columns we have
numerical_input = keras.layers.Input(shape=(len(num_col),), name='numerical_input')
# Create an input layer for each column tokenizer with nodes equal to our max sequence length
embedded_inputs = [keras.layers.Input(shape=(max_length,), name=f'embedded_input_{i}') for i in range(len(col_tokenizers))]
# Connect each text input layer to the embedding layer
embedded_sequences = [embed_input_layer(embed_input) for embed_input_layer, embed_input in zip(embedding_layers, embedded_inputs)]
print(f'Embedded sequences:')
for i in embedded_sequences:
    print(i)
# Concatenate them
concatenated_embedding = keras.layers.Concatenate()(embedded_sequences)
print(f'Concatenated embeddings: {concatenated_embedding}')
# LSTM layer for text processing
lstm_output = keras.layers.LSTM(64)(concatenated_embedding)

# Combine text and numerical features
all_features = keras.layers.Concatenate()([numerical_input, lstm_output])

# Output layers
output_delay = keras.layers.Dense(1, activation='relu', name='output_delay')(all_features)
output_length = keras.layers.Dense(1, activation='relu', name='output_length')(all_features)

# Define the model
model = keras.models.Model(inputs=[numerical_input] + embedded_inputs, outputs=[output_delay, output_length])
# Compile the model
model.compile(optimizer='adam', loss={'output_length': 'mean_squared_error', 'output_delay': 'mean_squared_error'})

# Get our data ready
#Split our numerical data
seed = 42
x_numerical_train, x_numerical_test, y_train, y_test = train_test_split(
    df[num_col].values,
    df[TARGET_COLUMNS],
    test_size=0.2,
    random_state=seed
)
# Split target variables
y_train_delay, y_train_length = y_train[TARGET_COLUMNS[0]], y_train[TARGET_COLUMNS[1]]
y_test_delay, y_test_length = y_test[TARGET_COLUMNS[0]], y_test[TARGET_COLUMNS[1]]
# Split the text data
x_text_train_list = []
x_text_test_list = []
for padded_sequence in padded_sequences_list:
    x_text_train, x_text_test = train_test_split(padded_sequences, test_size=0.2, random_state=seed)
    x_text_train_list.append(x_text_train)
    x_text_test_list.append(x_text_test)
# Should i standardize numerical data??

model.fit(
    [x_numerical_train] + x_text_train_list,
    {'output_delay': y_train_delay, 'output_length': y_train_length},
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

results = model.evaluate(
    [x_numerical_test] + x_text_test_list,
    {'output_delay': y_test_delay, 'output_length': y_test_length}
)

print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}, Test MSE: {results[2]}')