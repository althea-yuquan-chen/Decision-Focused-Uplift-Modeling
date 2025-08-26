import tensorflow as tf
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Multiply, Activation
from keras.optimizers import Adam, SGD
from keras.layers.core import Lambda, Dropout
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras

from keras.engine.topology import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


def get_roi_rank_model():
    feature_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-3))(feature_input)
    
    q_output =  Dense(1, activation="sigmoid", name="p1", kernel_regularizer=regularizers.l2(1e-3))(p1_hidden_1)

    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)
    
    
    qr = tf.math.log(q_output / (1 - q_output))
    qc = tf.math.log(1 - q_output)
    
    r_output = tf.reduce_sum(reward_input * qr * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * qr * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    c_output = tf.reduce_sum(cost_input * qc * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(cost_input * qc * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)

    loss = - (r_output + c_output)
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model


def get_direct_rank_model():
    feature_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(2e-3))(feature_input)
    
    q_output =  Dense(1, activation="tanh", name="p1", kernel_regularizer=regularizers.l2(2e-3))(p1_hidden_1)

    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)
    
    
    p_output = tf.exp(q_output) * treated_input / tf.reduce_sum(tf.exp(q_output) * treated_input) + tf.exp(q_output) * (1 - treated_input) / tf.reduce_sum(tf.exp(q_output) * (1 - treated_input))
    
    r_output = tf.reduce_sum(reward_input * p_output * (2 * treated_input - 1))
    c_output = tf.reduce_sum(cost_input * p_output * (2 * treated_input - 1))
    
    loss = c_output / r_output
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model


def get_roi_rank_criteo_model():
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    
    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(2.5e-5))(feature_input)
    
    q_output =  Dense(1, activation="sigmoid", name="p1", kernel_regularizer=regularizers.l2(2.5e-5))(p1_hidden_1)

    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)
    
    
    qr = tf.math.log(q_output / (1 - q_output))
    qc = tf.math.log(1 - q_output)
    
    r_output = tf.reduce_sum(reward_input * qr * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * qr * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
    c_output = tf.reduce_sum(cost_input * qc * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(cost_input * qc * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)

    loss = - (r_output + c_output)
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model



def get_direct_rank_criteo_model():
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-6))(feature_input)
   
    q_output =  Dense(1, activation="tanh", name="p1", kernel_regularizer=regularizers.l2(1e-6))(p1_hidden_1)

    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)
    
    
    p_output = tf.exp(q_output) * treated_input / tf.reduce_sum(tf.exp(q_output) * treated_input) + tf.exp(q_output) * (1 - treated_input) / tf.reduce_sum(tf.exp(q_output) * (1 - treated_input))
    
    r_output = tf.reduce_sum(reward_input * p_output * (2 * treated_input - 1))
    c_output = tf.reduce_sum(cost_input * p_output * (2 * treated_input - 1))
    
    loss = c_output / r_output
    
    final_model.add_loss(loss)
    final_model.add_metric(loss, name='obj')
    return final_model