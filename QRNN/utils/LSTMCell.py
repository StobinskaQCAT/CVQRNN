# Script taken from
# https://www.tensorflow.org/tutorials/structured_data/time_series
# For more explanation visit this website

# Model used here is called an autoregressive model. It first 'warms up'
# and then after the last input it produces output. The loss function 
# is calculated only after warmup so if we predict only 1 point it will
# be calculated on one point only.

import keras
import keras.backend as backend
import tensorflow as tf


class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, return_sequence=False):
        super().__init__()
        self.out_steps = out_steps
        self.return_sequence = return_sequence
        self.units = units
        # self.rnn_cell = MinimalRNNCell(units)
        self.rnn_cell = tf.keras.layers.LSTMCell(units)
        # self.rnn_cell = tf.keras.layers.GRUCell(units);

        # Also wrap the LSTMCell in an RNN to simplify the
        # `warmup` method.
        self.rnn = tf.keras.layers.RNN(self.rnn_cell,
                                       return_state=True,
                                       return_sequences=return_sequence)
        self.dense = tf.keras.layers.Dense(1)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.rnn(inputs)

        if self.return_sequence:
            return x, state

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def __call__(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        if not self.return_sequence:
            # Insert the first prediction.
            predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.rnn_cell(x, states=state,
                                     training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)

        if self.return_sequence:
            return prediction
        else:
            predictions = tf.transpose(predictions, [1, 0, 2])
            return predictions
