import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import LayerNormalization
from keras.metrics import RootMeanSquaredError 

# def build_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50),
#         Dropout(0.2),
#         Dense(1)
#     ])

#     model.compile(
#         optimizer="adam",
#         loss="mean_squared_error"
#     )

#     return model

def build_model(input_shape):
    model = Sequential([
        # Layer Normalization ajuda a manter a escala dos gradientes sob controle
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LayerNormalization(), 
        Dropout(0.2),
        
        LSTM(64),
        LayerNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'), # Camada intermediária para captar complexidade
        Dense(1) # Saída Linear para regressão
    ])
    
    # Adicionando clipnorm para evitar o erro 422 (gradientes explosivos)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[RootMeanSquaredError()])
    return model
    