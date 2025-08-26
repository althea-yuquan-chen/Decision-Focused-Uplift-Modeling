import tensorflow as tf
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Multiply, Activation
from keras.optimizers import Adam, SGD
from keras.layers import Lambda, Dropout
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras

#from keras.engine.topology import Layer
from keras.layers import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

def get_uplift_rank_model():
    feature_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input") 
    

    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(7e-4))(feature_input)
    
    p1_output =  Dense(1, name="p1", kernel_regularizer=regularizers.l2(7e-4))(p1_hidden_1)

    final_model = Model(inputs=[feature_input,treated_input, reward_input], outputs=p1_output)
    
    p_output = tf.exp(p1_output) / tf.reduce_sum(tf.exp(p1_output))
    
    q_output = tf.math.log(p_output)
    
    r_output = tf.reduce_sum(reward_input * q_output * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * q_output * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    
    loss = 0.0 - r_output
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model




def get_slearner_model():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")

    p1_input = concatenate([p0_input, treated_input])
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-4))(p1_input)
    p1_hidden_2 = Dense(32, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(1e-4))(
        p1_hidden_1)
    p1 = Dense(1, name="p1", kernel_regularizer=regularizers.l2(1e-4))(p1_hidden_2)

    final_model = Model(inputs=[p0_input, treated_input], outputs=p1)

    return final_model


def get_xlearner_model_tau0():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(3e-4))(p0_input)
    p1_hidden_2 = Dense(32, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(3e-4))(
        p1_hidden_1)
    p1 = Dense(1, name="p1", kernel_regularizer=regularizers.l2(3e-4))(p1_hidden_2)

    final_model = Model(inputs=p0_input, outputs=p1)

    return final_model


def get_xlearner_model():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(3e-4))(p0_input)
    p1_hidden_2 = Dense(32, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(3e-4))(
        p1_hidden_1)
    p1 = Dense(1, name="p1", kernel_regularizer=regularizers.l2(3e-4))(p1_hidden_2)

    final_model = Model(inputs=p0_input, outputs=p1)

    return final_model


#1e-5
def get_uplift_rank_criteo_model():
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input") 
    
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-5))(feature_input)
    
    p1_output =  Dense(1, name="p1", kernel_regularizer=regularizers.l2(1e-5))(p1_hidden_1)


    final_model = Model(inputs=[feature_input,treated_input, reward_input], outputs=p1_output)
    
    p_output = tf.exp(p1_output) / tf.reduce_sum(tf.exp(p1_output))
    
    q_output = tf.math.log(p_output)
    
    r_output = tf.reduce_sum(reward_input * q_output * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * q_output * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    
    loss = 0.0 - r_output
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model



def get_slearner_criteo_model():
    p0_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")

    p1_input = concatenate([p0_input, treated_input])
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-5))(p1_input)
    p1_hidden_2 = Dense(4, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(1e-5))(
        p1_hidden_1)
    p1 = Dense(1, activation="sigmoid", name="p1", kernel_regularizer=regularizers.l2(1e-5))(p1_hidden_2)

    final_model = Model(inputs=[p0_input, treated_input], outputs=p1)

    return final_model


def get_xlearner_criteo_model():
    p0_input = Input(shape=(12,), name="p0_raw_features")
    
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(5e-6))(p0_input)
    p1_hidden_2 = Dense(4, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(5e-6))(
        p1_hidden_1)
    p1 = Dense(1, name="p1", kernel_regularizer=regularizers.l2(5e-6))(p1_hidden_2)

    final_model = Model(inputs=p0_input, outputs=p1)

    return final_model