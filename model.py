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

# Tokenize and pad all text columns together
max_length = 30
tokenizer = Tokenizer(oov_token='<UNK>')
df = pd.read_csv('encoded_model_data.csv')
df.reset_index(drop=True, inplace=True)

# Combine text data from all columns
combined_text = df[TEXT_COLUMNS].apply(tuple, axis=1)
combined_text = combined_text.str.join(', ').values
print(combined_text[:5])

# Fit tokenizer to the combined text
tokenizer.fit_on_texts(combined_text)

# Tokenize and pad each text column individually
padded_sequences_list = []
vocab_size = len(tokenizer.word_index) + 1
for col in TEXT_COLUMNS:
    sequences = tokenizer.texts_to_sequences(df[col])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    padded_sequences_list.append(padded_sequences)

# Numerical columns
num_col = [col for col in df.columns if (col not in TEXT_COLUMNS) and ('Unnamed' not in col) and (col not in TARGET_COLUMNS)]
numerical_input = keras.layers.Input(shape=(len(num_col),), name='numerical_input')

# Textual input layers
embedded_inputs = [keras.layers.Input(shape=(max_length,), name=f'embedded_input_{i}') for i in range(len(TEXT_COLUMNS))]

# Embedding layer
embedding_dim = 100
embedding_layer = keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)

# Embed each text column
embedded_sequences = [embedding_layer(embed_input) for embed_input in embedded_inputs]

# Concatenate embedded sequences
concatenated_embedding = keras.layers.Concatenate()(embedded_sequences)

# LSTM layer
lstm_output = keras.layers.LSTM(64)(concatenated_embedding)
flatten = keras.layers.Flatten()(lstm_output)
text_dense1 = keras.layers.Dense(16)(flatten)
text_dense2 = keras.layers.Dense(5)(text_dense1)

# Combine text and numerical features
all_features = keras.layers.Concatenate()([numerical_input, text_dense2])

# Dense layers
dense1 = keras.layers.Dense(64, activation='relu', name='dense1')(all_features)
dense2 = keras.layers.Dense(128, activation='relu', name='dense2')(dense1)
dense3 = keras.layers.Dense(64, activation='relu', name='dense3')(dense2)
dense4 = keras.layers.Dense(32, activation='relu', name='dense4')(dense3)

# Output layers
output_delay = keras.layers.Dense(1, activation='linear', name='output_delay')(dense4)
output_length = keras.layers.Dense(1, activation='linear', name='output_length')(dense4)

# Define the model
model = keras.models.Model(inputs=[numerical_input] + embedded_inputs, outputs=[output_delay, output_length])

# Compile the model
model.compile(optimizer='adam', loss={'output_delay': 'mean_squared_error', 'output_length': 'mean_squared_error'})

# Prepare data for training
seed = 42
x_numerical_train, x_numerical_test, y_train, y_test = train_test_split(
    df[num_col].values,
    df[TARGET_COLUMNS],
    test_size=0.2,
    random_state=seed
)
y_train_delay, y_train_length = y_train[TARGET_COLUMNS[0]], y_train[TARGET_COLUMNS[1]]
y_test_delay, y_test_length = y_test[TARGET_COLUMNS[0]], y_test[TARGET_COLUMNS[1]]

#Standardize data
scaler = StandardScaler()
x_numerical_train = scaler.fit_transform(x_numerical_train)
x_numerical_test = scaler.fit_transform(x_numerical_test)

x_text_train_list = []
x_text_test_list = []

for padded_sequence in padded_sequences_list:
    x_text_train, x_text_test = train_test_split(padded_sequence, test_size=0.2, random_state=seed)
    x_text_train_list.append(x_text_train)
    x_text_test_list.append(x_text_test)

model.fit(
    [x_numerical_train] + x_text_train_list,
    {'output_delay': y_train_delay, 'output_length': y_train_length},
    epochs=15,
    batch_size=32,
    validation_split=0.2
)

results = model.evaluate(
    [x_numerical_test] + x_text_test_list,
    {'output_delay': y_test_delay, 'output_length': y_test_length}
)

print(f'Test Loss: {results}')

