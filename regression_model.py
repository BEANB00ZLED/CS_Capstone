import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras
import keras_tuner as kt
from util import TEXT_COLUMNS, TARGET_COLUMNS, save_model, get_model_info
import os
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

FILE = 'encoded_model_data.csv'
DIR = 'test_dir'

# Tokenize and pad all text columns together
max_length = 30
tokenizer = Tokenizer(oov_token='<UNK>')
df = pd.read_csv(FILE)
df.reset_index(drop=True, inplace=True)

# Combine text data from all columns
combined_text = df[TEXT_COLUMNS].apply(tuple, axis=1)
combined_text = combined_text.str.join(', ').values

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

# Prepare data for training
seed = 87
test_split = 0.2
x_numerical_train, x_numerical_test, y_train, y_test = train_test_split(
    df[num_col].values,
    df[TARGET_COLUMNS],
    test_size=test_split,
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
    x_text_train, x_text_test = train_test_split(padded_sequence, test_size=test_split, random_state=seed)
    x_text_train_list.append(x_text_train)
    x_text_test_list.append(x_text_test)

def get_optimized_model() -> keras.Model:
    def build_model(hp):
        numerical_input = keras.layers.Input(shape=(len(num_col),), name='numerical_input')

        # Textual input layers
        embedded_inputs = [keras.layers.Input(shape=(max_length,), name=f'embedded_input_{i}') for i in range(len(TEXT_COLUMNS))]

        # Embedding layer
        embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
        embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)

        # Embed each text column
        embedded_sequences = [embedding_layer(embed_input) for embed_input in embedded_inputs]

        # Concatenate embedded sequences
        concatenated_embedding = keras.layers.Concatenate()(embedded_sequences)

        # LSTM layer
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        lstm_output = keras.layers.LSTM(lstm_units, name='lstm_output')(concatenated_embedding)
        flatten = keras.layers.Flatten()(lstm_output)
        text_dense1 = keras.layers.Dense(hp.Int('text_dense1_units', min_value=16, max_value=64, step=16), activation='relu', name='text_dense1')(flatten)
        text_dense2 = keras.layers.Dense(hp.Int('text_dense2_units', min_value=4, max_value=32, step=4), activation='relu', name='text_dense2')(text_dense1)

        # Combine text and numerical features
        all_features = keras.layers.Concatenate()([numerical_input, text_dense2])

        # Dense layers
        dense1 = keras.layers.Dense(hp.Int('dense1_units', min_value=64, max_value=256, step=64), activation='relu', name='dense1')(all_features)
        dense2 = keras.layers.Dense(hp.Int('dense2_units', min_value=64, max_value=256, step=64), activation='relu', name='dense2')(dense1)
        dense3 = keras.layers.Dense(hp.Int('dense3_units', min_value=64, max_value=128, step=32), activation='relu', name='dense3')(dense2)
        dense4 = keras.layers.Dense(hp.Int('dense4_units', min_value=16, max_value=128, step=16), activation='relu', name='dense4')(dense3)

        # Output layers
        output_delay = keras.layers.Dense(1, activation='linear', name='output_delay')(dense4)
        output_length = keras.layers.Dense(1, activation='linear', name='output_length')(dense4)

        # Define the model
        model = keras.models.Model(inputs=[numerical_input] + embedded_inputs, outputs=[output_delay, output_length])

        # Determine optimal optimizer setup
        optimizer = keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
        
        # Compile the model
        model.compile(
            optimizer=optimizer, 
            loss={
                'output_delay': 'mean_squared_error',
                'output_length': 'mean_squared_error'
            },
            metrics={
                'output_delay':[
                    keras.metrics.MeanSquaredError(name='mse_delay'), 
                    keras.metrics.RootMeanSquaredError(name='rmse_delay'),
                    keras.metrics.MeanAbsoluteError(name='mae_delay'),
                ],
                'output_length':[
                    keras.metrics.MeanSquaredError(name='mse_length'),
                    keras.metrics.RootMeanSquaredError(name='rmse_length'),
                    keras.metrics.MeanAbsoluteError(name='mae_length')
                ]
            }
        )

        return model
    
'''def get_optimized_model_v2() -> keras.Model:
    

    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_loss',
        max_epochs=60,
        factor=3,
        directory='results_dir',
        project_name=DIR
    )
    tuner.search(
        [x_numerical_train] + x_text_train_list,
        {'output_delay': y_train_delay, 'output_length': y_train_length},
        epochs=30,
        validation_split=0.2
    )
    model = tuner.get_best_models(num_models=1)[0]
    return model'''


def main():
    model = keras.models.load_model('regression_model.keras')
    
    results = model.evaluate(
        [x_numerical_test] + x_text_test_list,
        {'output_delay': y_test_delay, 'output_length': y_test_length}
    )
    
    #Computer r2 scores for delay and length separately
    y_true = [[i, j] for i, j in zip(y_test_delay.values, y_test_length.values)]
    y_pred = model.predict([x_numerical_test] + x_text_test_list)
    r2_delay = r2_score(y_test_delay, y_pred[0])
    r2_length = r2_score(y_test_length, y_pred[1])

    mape_delay = mean_absolute_percentage_error(y_test_delay, y_pred[0])
    mape_length = mean_absolute_percentage_error(y_test_length, y_pred[1])

    # Quick and dirty metrics for the paper
    print(f'MAPE FOR MODEL\nDelay: {mape_delay}\nLength: {mape_length}')
    delay_under = y_pred[0].flatten() < y_test_delay
    delay_under = round(np.mean(delay_under) * 100, 2)
    print(f'DELAY predicts less than actual {delay_under}% of the time and over the actual {100 - delay_under}% of the time')
    length_under = y_pred[1].flatten() < y_test_length
    length_under = round(np.mean(length_under) * 100, 2)
    print(f'LENGTH predicts less than actual {length_under}% of the time and over the acual {100 - length_under}% of the time')

    #Graphing pred vs true
    delay_graph = sns.scatterplot(
        x=y_test_delay,
        y=y_pred[0].flatten(),
        color='red',
        marker='o'
    )
    theor = np.linspace(0, 10)
    plt.plot(theor, theor, linestyle='-', color='black')
    plt.grid(True)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    delay_graph.set(
        title='Actual Versus Predicted Delay',
        ylabel='Predicted Delay [minutes]',
        xlabel='Actual Delay [minutes]'
    )
    plt.show()

    plt.close()

    #Graphing pred vs true
    length_graph = sns.scatterplot(
        x=y_test_length,
        y=y_pred[1].flatten(),
        color='red',
        marker='o'
    )
    theor = np.linspace(0, 11)
    plt.plot(theor, theor, linestyle='-', color='black')
    plt.grid(True)
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    length_graph.set(
        title='Actual Versus Predicted Length',
        ylabel='Predicted Length [miles]',
        xlabel='Actual Length [miles]'
    )
    plt.show()


    print(f'Test Loss: {results}\nR^2 values are\nDelay: {r2_delay}\nLength: {r2_length}')
    #save_model(model, results, r2_delay, r2_length)

if __name__ == '__main__':
    main()
