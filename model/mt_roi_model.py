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



def get_mt_slearner_model():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")

    p1_input = concatenate([p0_input, treated_input])
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-4))(p1_input)
    p1_hidden_2 = Dense(32, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(1e-4))(
        p1_hidden_1)
    p1 = Dense(1, name="p1", kernel_regularizer=regularizers.l2(1e-4))(p1_hidden_2)

    final_model = Model(inputs=[p0_input, treated_input], outputs=p1)

    return final_model



def l2_reg(weight_matrix):
    return 0.03 * tf.reduce_sum(tf.square(weight_matrix[:76,:])) + 0.001 * tf.reduce_sum(tf.square(weight_matrix[76,:]))

# This version requires that the count of samples in each treatment is the same or close
def get_rank_marginal_utility_model():
    
    feature_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    p1_input = concatenate([feature_input, treated_input])
    
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=l2_reg)
    
    p1_output =  Dense(1, activation="sigmoid", name="p1", kernel_regularizer=regularizers.l2(1e-2))
    
    
    qij = p1_output(p1_hidden_1(p1_input))
    
    final_model = Model(inputs=[feature_input,treated_input, reward_input, cost_input], outputs=qij)
    
    qij_square = tf.square(qij)
    
    
    p1_minus_one_input = concatenate([feature_input, treated_input - 0.1])
    
    qij_minus_one = p1_output(p1_hidden_1(p1_minus_one_input))
    
    qij_minus_one_square = tf.square(qij_minus_one)
    
    
    mask_1 = tf.cast(tf.greater(treated_input,tf.constant([0.0])), tf.float32)
    
    mask_2 = tf.cast(tf.less(treated_input,tf.constant([0.6])), tf.float32)
    
    r_norm = qij_minus_one * mask_1 - qij * mask_2
    
    c_norm = qij_minus_one_square * mask_1 - qij_square * mask_2

    
    r_output = tf.reduce_mean(reward_input * r_norm)
    c_output = tf.reduce_mean(cost_input * c_norm)
    
    loss = c_output - r_output
    
    final_model.add_loss(loss * 1000)
    final_model.add_metric(loss * 1000, name='obj')
    return final_model


def get_mt_slearner_model():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")

    p1_input = concatenate([p0_input, treated_input])
    p1_hidden_1 = Dense(64, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-4))(p1_input)
    p1_hidden_2 = Dense(32, activation="relu", name="p1_hidden_2", kernel_regularizer=regularizers.l2(1e-4))(
        p1_hidden_1)
    p1 = Dense(1, name="p1", kernel_regularizer=regularizers.l2(1e-4))(p1_hidden_2)

    final_model = Model(inputs=[p0_input, treated_input], outputs=p1)

    return final_model


def calcu_price_elastic_2(inputs):
    p0 = inputs[0]
    cost0 = inputs[1]
    alpha = inputs[2]
    cost = inputs[3]
    return 1.0 - (1.0 - p0) * K.exp(-1 * alpha * (cost - cost0))


def get_price_elastic_curve_model_2():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")

    
    p0_hidden_1 = Dense(64, activation="relu", name="p0_hidden_1", kernel_regularizer=regularizers.l2(1e-4))(p0_input)
    p0_hidden_2 = Dense(32, activation="relu", name="p0_hidden_2", kernel_regularizer=regularizers.l2(1e-4))(p0_hidden_1)
    p0 = Dense(1, activation="sigmoid", name="p0", kernel_regularizer=regularizers.l2(1e-4))(p0_hidden_2)
    
    
    alpha_hidden_1 = Dense(64, activation="relu", name="alpha_hidden_1", kernel_regularizer=regularizers.l2(5e-5))(
        p0_input)
    alpha_hidden_2 = Dense(32, activation="relu", name="alpha_hidden_2", kernel_regularizer=regularizers.l2(5e-5))(
        alpha_hidden_1)
    alpha = Dense(1, name="alpha", kernel_regularizer=regularizers.l2(5e-5))(alpha_hidden_2)
    
    
    
    cost_input = concatenate([p0_input, treated_input])
    cost_hidden_1 = Dense(64, activation="relu", name="cost_hidden_1", kernel_regularizer=regularizers.l2(1e-4))
    cost_hidden_2 = Dense(32, activation="relu", name="cost_hidden_2", kernel_regularizer=regularizers.l2(1e-4))
    cost_output = Dense(1, name="cost_output", kernel_regularizer=regularizers.l2(1e-4))
    
    cost = cost_output(cost_hidden_2(cost_hidden_1(cost_input)))
    
    cost0_input = concatenate([p0_input, treated_input - treated_input])
    cost0 = cost_output(cost_hidden_2(cost_hidden_1(cost0_input)))
    
    
    p1 = Lambda(calcu_price_elastic, name="p1_output")([p0, cost0, alpha, cost])
    
    final_model = Model(inputs=[p0_input, treated_input], outputs=[p1, cost])

    return final_model


def calcu_price_elastic(inputs):
    p0 = inputs[0]
    alpha = inputs[1]
    cost = inputs[2]
    return 1.0 - (1.0 - p0) * K.exp(-1 * alpha * cost)


def get_price_elastic_curve_model():
    p0_input = Input(shape=(76,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")

    
    p0_hidden_1 = Dense(64, activation="relu", name="p0_hidden_1", kernel_regularizer=regularizers.l2(1e-4))(p0_input)
    p0_hidden_2 = Dense(32, activation="relu", name="p0_hidden_2", kernel_regularizer=regularizers.l2(1e-4))(p0_hidden_1)
    p0 = Dense(1, activation="sigmoid", name="p0", kernel_regularizer=regularizers.l2(1e-4))(p0_hidden_2)
    
    
    alpha_hidden_1 = Dense(64, activation="relu", name="alpha_hidden_1", kernel_regularizer=regularizers.l2(5e-5))(
        p0_input)
    alpha_hidden_2 = Dense(32, activation="relu", name="alpha_hidden_2", kernel_regularizer=regularizers.l2(5e-5))(
        alpha_hidden_1)
    alpha = Dense(1, name="alpha", kernel_regularizer=regularizers.l2(5e-5))(alpha_hidden_2)
    
    
    
    cost_input = concatenate([p0_input, treated_input])
    cost_hidden_1 = Dense(64, activation="relu", name="cost_hidden_1", kernel_regularizer=regularizers.l2(1e-4))(cost_input)
    cost_hidden_2 = Dense(32, activation="relu", name="cost_hidden_2", kernel_regularizer=regularizers.l2(1e-4))(cost_hidden_1)
    cost = Dense(1, activation="sigmoid", name="cost", kernel_regularizer=regularizers.l2(1e-4))(cost_hidden_2)
    
    p1 = Lambda(calcu_price_elastic, name="p1_output")([p0, alpha, cost])
    
    final_model = Model(inputs=[p0_input, treated_input], outputs=[p1, cost])

    return final_model


