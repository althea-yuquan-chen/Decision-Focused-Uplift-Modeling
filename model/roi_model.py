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

from keras.layers import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


# Keras 3's Functional API removed support for calling `Model.add_loss()`
# with a tensor built directly on `Input(...)` outputs (the pattern the
# original Keras 2 code used) — the loss must instead be computed inside a
# Layer's `call()`. The two Layer classes below reproduce the exact original
# math for ROI Rank / Direct Rank and pass `q_output` through unchanged, so
# they're drop-in replacements.

class _RoiRankLoss(Layer):
    def call(self, inputs):
        q_output, treated_input, reward_input, cost_input = inputs

        qr = tf.math.log(q_output / (1 - q_output))
        qc = tf.math.log(1 - q_output)

        r_output = tf.reduce_sum(reward_input * qr * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(reward_input * qr * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)
        c_output = tf.reduce_sum(cost_input * qc * treated_input) / tf.reduce_sum(treated_input) - tf.reduce_sum(cost_input * qc * (1 - treated_input)) / tf.reduce_sum(1 - treated_input)

        loss = - (r_output + c_output)

        self.add_loss(loss)
        return q_output


class _DirectRankLoss(Layer):
    def call(self, inputs):
        q_output, treated_input, reward_input, cost_input = inputs

        p_output = tf.exp(q_output) * treated_input / tf.reduce_sum(tf.exp(q_output) * treated_input) + tf.exp(q_output) * (1 - treated_input) / tf.reduce_sum(tf.exp(q_output) * (1 - treated_input))

        r_output = tf.reduce_sum(reward_input * p_output * (2 * treated_input - 1))
        c_output = tf.reduce_sum(cost_input * p_output * (2 * treated_input - 1))

        loss = c_output / r_output

        self.add_loss(loss)
        return q_output


def get_roi_rank_criteo_model():
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(2.5e-5))(feature_input)

    q_output =  Dense(1, activation="sigmoid", name="p1", kernel_regularizer=regularizers.l2(2.5e-5))(p1_hidden_1)

    q_output = _RoiRankLoss(name="roi_rank_loss")([q_output, treated_input, reward_input, cost_input])

    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)

    return final_model


def get_direct_rank_criteo_model():
    feature_input = Input(shape=(12,), name="p0_raw_features")
    treated_input = Input(shape=(1,), name="treated_input")
    reward_input = Input(shape=(1,), name="reward_input")
    cost_input = Input(shape=(1,), name="cost_input")

    p1_hidden_1 = Dense(8, activation="relu", name="p1_hidden_1", kernel_regularizer=regularizers.l2(1e-6))(feature_input)

    q_output =  Dense(1, activation="tanh", name="p1", kernel_regularizer=regularizers.l2(1e-6))(p1_hidden_1)

    q_output = _DirectRankLoss(name="direct_rank_loss")([q_output, treated_input, reward_input, cost_input])

    final_model = Model(inputs=[feature_input, treated_input, reward_input, cost_input], outputs=q_output)

    return final_model