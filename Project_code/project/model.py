import os
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import joblib
from tensorflow.keras.layers import Input, Dense, GRU, Embedding


latent_dim = 512
num_encoder_tokens = 4096
num_decoder_tokens = 1500
time_steps_encoder = 80
max_probability = -1
save_model_path = 'model_final'




def connect_decoder(transfer_values):
 # Map the transfer-values so the dimensionality matches
 # the internal state of the GRU layers. This means
 # we can use the mapped transfer-values as the initial state
 # of the GRU layers.
 initial_state = decoder_transfer_map(transfer_values)
 # Start the decoder-network with its input-layer.
 net = decoder_input

 # Connect the embedding-layer.
 net = decoder_embedding(net)

 # Connect all the GRU layers.
 net = decoder_gru1(net, initial_state=initial_state)
 net = decoder_gru2(net, initial_state=initial_state)
 net = decoder_gru3(net, initial_state=initial_state)
 # Connect the final dense layer that converts to
 # one-hot encoded arrays.
 decoder_output = decoder_dense(net)
 return decoder_output

state_size = 512
embedding_size = 128
decoder_transfer_map = Dense(state_size,
activation='tanh',
name='decoder_transfer_map')
decoder_input = Input(shape=(None, ), name='decoder_input')
num_words=10
decoder_gru1 = GRU(state_size, name='decoder_gru1',return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',return_sequences=True)
decoder_dense = Dense(num_words,activation='softmax',name='decoder_output')



def inference_model():
    """Returns the model that will be used for inference"""
    with open(os.path.join(save_model_path, 'tokenizer' + str(num_decoder_tokens)), 'rb') as file:
        tokenizer = joblib.load(file)
    # loading encoder model. This remains the same
    inf_encoder_model = load_model(os.path.join(save_model_path, 'encoder_model.h5'))
    
    # inference decoder model loading
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    inf_decoder_model.load_weights(os.path.join(save_model_path, 'decoder_model_weights.h5'))
    return tokenizer, inf_encoder_model, inf_decoder_model


