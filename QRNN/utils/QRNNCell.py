import numpy as np
import strawberryfields as sf
import tensorflow as tf
from strawberryfields import ops


class QRNN:
    """
    Class for one layer or Qunatum Reccurant Neural Network. It takes
    both hidden state and input data and output hidden state and
    output data, so it can work in loop.
    """

    def __init__(self, in_modes, hid_modes, cutoff_dim, nonlinear=0, layers=1,
                 repetitions=1, energy_transmissivity=1):
        """
        Initialization of QRNN

        :param in_modes: int, number of input modes
        :param hid_modes: int, number of hidden (history) modes
        :param cutoff_dim: networks cutoff dimension for the Hilbert space
        :param nonlinear: int, type of non-linearity:
            - use[1] / not use[0] kerr or
            - PNR measureemnet [2] 
            - single photon encdoing with PNR [3]
        :param layers: number of layers between input and output
        :param repetitions: number of repetitions of all layers
            (not new weights being introduced)
        :param energy_transmissivity: channel loss param (0 - lossless,
        1 - total loss)
        """
        #
        #
        self.in_modes = in_modes
        self.hid_modes = hid_modes
        self.modes = self.in_modes + self.hid_modes
        self.cutoff_dim = cutoff_dim
        self.nonlinear = nonlinear
        self.layers = layers
        self.repetitions = repetitions
        self.energy_transmissivity = energy_transmissivity

        # mask for coloring active and passive parameters differently
        self.color_mask = []
        self.weights, self.scale = self.init_weights()
        #  get names for parameters to pass them in TF
        self.params_names = self.get_params_names(sf.Program(
            self.modes))  #

    def __call__(self, x, prev_state, final=False):
        """
        One layer action

        :param x: np.array(batch_size) | tf.tensor(batch_size), input data
        :param prev_state: sf state, state of the previous layer

        :return: tuple(float, sf density matrix), rescaled mean number of
            photons on the 0th mode and state of all modes
            (hidden state after measurement of 0th mode)
        """

        # batch size different from 1 does not work great,
        # I don't know exactly how, but it is like it is
        if x.shape == () or x.shape[0] == 1:
            batch_size = None
        else:
            batch_size = x.shape[0]

        eng = sf.Engine(backend="tf",
                        backend_options={"cutoff_dim": self.cutoff_dim,
                                         "batch_size": batch_size})

        # from previous_state obtain the reduced matrix of history state,
        # so all modes which are not measured in the process
        if prev_state is not None:
            # reduced matrix
            # dm = prev_state.reduced_dm(
            #     modes=[i for i in range(self.in_modes, self.modes)])

            # full matrix; later need to measure the modes
            dm = prev_state.dm()
        else:
            dm = None

        # run the circuit and store the result
        state = eng.run(
            self.qrnn_cell(dm),
            args=self.wrap_params(dm, x, self.weights)
        ).state

        if final is False:  # measure "IO" modes
            to_measure = range(self.in_modes)
        else:  # measure "history" modes
            to_measure = range(self.in_modes, self.modes)

        ########################################
        # Process the output (few possible ways)
        ########################################
        # 1) store the output of mean_photon for zeroth mode
        # std is just in case for new regularization
        #
        # mean_x, std_x = state.mean_photon(0)
        # mean_x = self.scale * mean_x

        # 2) what can be also measured (and what is done in the paper) is the
        # homodyne measurement so there are two options
        #
        # mean_x, std_x = state.quad_expectation(0)
        # mean_x = self.scale * mean_x

        # 3) just in case we want to use all measurements i.e. for using dense
        # layer at the end of quantum circuit
        #
        # for i in range(self.in_modes - 1):
        #     mean_x = np.vstack((mean_x, state.mean_photon(i)[0]))
        #     std_x = np.vstack((std_x, state.mean_photon(i)[1]))

        # 4) the output is the mean of the observables' means  multiplied by the
        # trainable parameter [Takaki2021]
        #
        mean_x = 0

        for i in to_measure:
            if self.nonlinear in [2, 3]:
                mean_x += state.mean_photon(i)[0]
            else:
                mean_x += state.quad_expectation(i)[0]
        mean_x /= len(to_measure)

        # scale the output which is also trainable
        mean_x *= self.scale

        return mean_x, state

    def init_weights(self, active_sd=0.01, passive_sd=0.1):
        """
        Initialize a 2D TensorFlow Variable containing
        normally-distributed random weights for an ``N`` mode quantum
        neural network with 1 layer.
        Taken from
        https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html

        The tutorial recommend to use tiny starting value for active
        gates, like 0.0001, but then it jumps straight to about 0.05,
        so I think it is more reasonable to start at slightly higher
        value. BUT: the higher you start the more trace you loose

        :param active_sd: float, the standard deviation used when
                initializing the normally-distributed weights for the
                active parameters (displacement, squeezing, and Kerr
                magnitude)
        :param passive_sd: float, the standard deviation used when
                initializing the normally-distributed weights for the
                passive parameters (beamsplitter angles and all gate
                phases)

        :return: tf.Variable[tf.float32], A TensorFlow Variable of shape
                ``[2*(max(1, modes-1) + modes**2 + modes)]``.:
        """

        # Number of interferometer parameters:
        M = int(self.modes * (self.modes - 1)) + max(1, self.modes - 1)

        # Create the TensorFlow variables
        int1_weights = tf.random.normal(shape=[self.layers, M],
                                        stddev=passive_sd)
        s_weights = tf.random.normal(shape=[self.layers, self.modes],
                                     stddev=active_sd)
        int2_weights = tf.random.normal(shape=[self.layers, M],
                                        stddev=passive_sd)
        dr_weights = tf.random.normal(shape=[self.layers, self.modes],
                                      stddev=active_sd)
        dp_weights = tf.random.normal(shape=[self.layers, self.modes],
                                      stddev=passive_sd)

        weights = tf.concat(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights],
            axis=1)

        # if nonlinear==1 then add kerr gates weights as trainable
        if self.nonlinear == 1:
            k_weights = tf.random.normal(shape=[self.layers, self.modes],
                                         stddev=active_sd)
            weights = tf.concat(
                [weights, k_weights], axis=1)

        weights = tf.Variable(weights)
        scale = tf.Variable(1.)

        # create the color mask to highlight different types of
        # parameters while plotting
        self.color_mask = [0] * M + \
                          [1] * self.modes + \
                          [0] * M + \
                          [1] * self.modes + \
                          [0] * self.modes + \
                          [1] * self.modes

        self.color_mask = self.color_mask * self.layers

        # print("Number of free parameters: ", np.prod(weights.shape))

        return weights, scale

    def interferometer(self, params, q):
        """
        Parameterised interferometer acting on ``N`` modes.  Taken from
        https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html

        :param params: (list[float]), list of length
            ``max(1, N-1) + (N-1)*N`` parameters.
            * The first ``N(N-1)/2`` parameters correspond to the
                beamsplitter angles
            * The second ``N(N-1)/2`` parameters correspond to the
                beamsplitter phases
            * The final ``N-1`` parameters correspond to local rotation
                on the first N-1 modes
        :param q: (list[RegRef]) list of Strawberry Fields quantum
            registers the interferometer is to be applied to

        :return: -
        """

        N = len(q)
        theta = params[:N * (N - 1) // 2]
        phi = params[N * (N - 1) // 2:N * (N - 1)]
        rphi = params[-N + 1:]

        if N == 1:
            # the interferometer is a single rotation
            ops.Rgate(rphi[0]) | q[0]
            return

        n = 0  # keep track of free parameters

        # Apply the rectangular beamsplitter array
        # The array depth is N
        for j in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                # skip even or odd pairs depending on layer
                if (j + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1

        # apply the final local phase shifts to all modes except the last one
        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]

    def layer(self, params, q):
        """
        CV quantum neural network layer acting on ``N`` modes. Taken from
        https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html


        :param params: (list[float]), list of weights
        :param q: (list[RegRef]), list of Strawberry Fields quantum
            registers the layer is to be applied to

        :return: -
        """

        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)

        int1 = params[:M]
        s = params[M:M + N]
        int2 = params[M + N:2 * M + N]
        dr = params[2 * M + N:2 * M + 2 * N]
        dp = params[2 * M + 2 * N:2 * M + 3 * N]
        k = params[2 * M + 3 * N:2 * M + 4 * N]

        # begin layer
        self.interferometer(int1, q)

        for i in range(N):
            # squeeze all the modes
            ops.Sgate(s[i]) | q[i]

        self.interferometer(int2, q)

        for i in range(N):
            # displace all the modes
            ops.Dgate(dr[i], dp[i]) | q[i]
            if self.nonlinear == 1:
                # apply Kerr if requested
                ops.Kgate(k[i]) | q[i]

    def qrnn_cell(self, previous_state=None):
        """
        Create one layer of QRNN. In the circuit there is
        **no measurement**, but it is obtained from the density matrix
        of the state, and for the next layer reduced matrix is inputted
        ===
        Change:
        Create one layer of QRNN. It starts with inputing the previous 
        density matrix, then performing a measuremnt (which by default
        sets the measured modes to |0>), then data input, then layer and
        at the end add of loss. It returns the program which then is run
        at at the end we obtain the state where we can get our "measurement"
        by just 'state.mean_photon()' or similar.

        :param previous_state: sf.dm, density matrix of the previous
            state, if any (default=None)

        :return: sf.Program(),  with implemented layer
        """

        prog = sf.Program(self.modes)
        params = self.get_params_names(prog)

        with prog.context as q:

            if previous_state is not None:
                # input the previous state (full matrix)
                ops.DensityMatrix(prog.params("previous_state")) | q
                for i in range(self.in_modes):
                    if self.nonlinear in [2, 3]:
                        # measure in Fock state
                        ops.MeasureFock() | q[i]
                    else:
                        # measure the output modes (homodyne) 
                        ops.MeasureX | q[i]

            # in this way does not make any sense
            # if self.nonlinear == 2:
            #     # add nonlinearity at the beginning
            #     for k in range(self.in_modes):
            #         ops.Fock(1) | q[k]

            for i in range(self.repetitions):
                for j in range(self.layers):
                    for k in range(self.in_modes):
                        if self.nonlinear == 3:
                            # single photon encoding
                            ops.Ket(prog.params("input_data")) | q[k]
                        else:
                            # amplitude encoding of the data
                            ops.Dgate(prog.params("input_data")) | q[k]
                            # angle encoding of the data
                            # ops.Dgate(1, np.pi * prog.params("input_data")) | q[k]
                            # squeeze encoding
                            # ops.Sgate(prog.params("input_data")) | q[k]
                    self.layer(params[j], q)

            for i in range(self.modes):
                # add some noise to each of the wire
                ops.LossChannel(self.energy_transmissivity) | q[i]

        return prog

    def wrap_params(self, previous_state, input_data, weights):
        """
        Wrap numerical parameters into the tensorflow tensors

        :param previous_state: sf density matrix, previous density
            matrix of not measured modes
        :param input_data: float, input datapoint (only one at the time)
        :param weights: list(float), current weights

        :return: dict, of names and values of the parameters of the
            network
        """

        mapping = {p.name: w for p, w in
                   zip(self.params_names.flatten(), tf.reshape(weights, [-1]))}

        # --------------------------------------------------------------
        # Used for qubit encoding |psi> = \alpha |0> + \beta |1>
        #
        if self.nonlinear == 3:
            if input_data.shape == () or input_data.shape[0] == 1:
                psi_np = np.zeros([self.cutoff_dim], dtype=np.complex128)
                # input data as the amplitude of the vaccum state
                psi_np[0] = input_data
                # set up the |1> amplitude so it is a proper state
                psi_np[1] = np.sqrt(1 - input_data ** 2)
            else:
                psi_np = np.zeros([input_data.shape[0], self.cutoff_dim],
                                  dtype=np.complex128)
                # input data as the amplitude of the vaccum state
                psi_np[:, 0] = input_data
                # set up the |1> amplitude so it is a proper state
                psi_np[:, 1] = np.sqrt(1 - input_data ** 2)

            psi_tf = tf.convert_to_tensor(psi_np, dtype=tf.complex128)

            data = {"input_data": psi_tf}
        # --------------------------------------------------------------

        else:
            data = {"input_data": input_data}

        if previous_state is not None:
            state = {"previous_state": previous_state}
            return {**mapping, **data, **state}

        return {**mapping, **data}

    def get_params_names(self, prog):
        """
        Create and get the names of the parameters

        :param prog: sf.Program(), program which we take parameters of

        :return: list(string), list of names of all the free parameters
            in the program
        """
        num_params = np.prod(self.weights.shape)
        sf_params = np.arange(num_params).reshape(
            self.weights.shape).astype(str)
        sf_params = np.array([prog.params(*i) for i in sf_params])

        return sf_params


# ----------------------------------------------------------------------
# Evaluate QRNN
# ----------------------------------------------------------------------

def get_pred(qrnn, function, in_len, out_len):
    """
    Getting the prediction of the trained network for all available data
    points -- both training and validation. Used for plotting

    :param qrnn: class(QRNN), trained network
    :param function: np.array(), dataset used for training
    :param in_len: int, number of input data points
    :param out_len: int, number of output data

    :return: np.array(), output of the network for all data set
    """

    prediction = np.array([])

    for i in range(0, len(function) - in_len, out_len):
        state = None
        result = 0

        for x in function[i:i + in_len]:
            result, state = qrnn(x, state)

        prediction = np.append(prediction, result)

        for _ in range(out_len - 1):
            result, state = qrnn(result, state)
            prediction = np.append(prediction, result)

        print(
            "\r{:.1f}% of samples".format(100 * i / (len(function) - in_len)),
            end="")

    return prediction


def get_fore(qrnn, xs, length, out_len):
    """
    Forecasting of the network
    :param qrnn: class(QRNN), trained network
    :param xs: np.array(), input data from which network will forecast
    :param length: int, how many points to forecast in total
    :param out_len: int, how many points to forecast at one run

    :return: np.array(), forecasting of len = length // out_len * out_len
    """

    forecast = np.array([])

    for i in range(length // out_len):
        state = None
        mean_x = 0

        # remember input set ...
        for x in xs:
            mean_x, state = qrnn(x, state)

        # ... first output ...
        xs = np.concatenate((xs[1:], [mean_x]))
        forecast = np.append(forecast, mean_x)

        # ... and try to predict next 'out_len' states
        for _ in range(out_len - 1):
            mean_x, state = qrnn(mean_x, state)

            # remember the prediction
            xs = np.concatenate((xs[1:], [mean_x]))
            forecast = np.append(forecast, mean_x)

        print("\r{:.1f}% of samples".format(100 * i / (length // out_len)),
              end="")

    return forecast


def get_pred_copy(qrnn, xs):
    """
    Prediction of the network for copying task
    :param qrnn: class(QRNN), trained network
    :param xs: np.array(), input data from which network will forecast

    :return: np.array(), forecasting of len = len(xs)
    """

    forecast = []
    state = None
    mean_x = 0

    # remember input set ...
    for x in xs:
        mean_x, state = qrnn(x, state)
        forecast.append(mean_x)

    return forecast
