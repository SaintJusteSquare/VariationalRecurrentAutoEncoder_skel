import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import variable, zeros, concatenate
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRUCell, GRU


class Sampling(layers.Layer):
    """Uses (z_mean, z_std) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + z_std * epsilon

    def sample_sequence(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        timesteps = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, timesteps, dim))
        return z_mean + z_std * epsilon


class GRULayer(layers.Layer):

    def __init__(self, hdim, **kwargs):
        super().__init__(**kwargs)
        self.GRU1 = GRUCell(hdim, name='rnn1')
        self.GRU2 = GRUCell(hdim, name='rnn2')
        self.GRU3 = GRUCell(hdim, name='rnn3')

    def call(self, inputs, states):
        o1, s1 = self.GRU1(inputs, states)
        o2, s2 = self.GRU2(o1, s1)
        outputs, s3 = self.GRU3(o2, s2)
        return outputs, s3

    def get_initial_state(self, inputs):
        h1 = self.GRU1.get_initial_state(inputs=inputs)
        return h1

    def reset_state(self, inputs):
        batch = inputs.shape[0]
        H = zeros((batch, h_dim))
        self.GRU1.__setstate__(H)
        self.GRU2.__setstate__(H)
        self.GRU3.__setstate__(H)


class Vrnn(tf.keras.Model):

    def __init__(self, x_dim, x2s_dim, h_dim, z_dim, z2s_dim, q_z_dim, p_z_dim, p_x_dim, mode='gauss', k=1):
        super(Vrnn, self).__init__()
        self.x_dim = x_dim
        self.x2s_dim = x2s_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.z2s_dim = z2s_dim

        self.q_z_dim = q_z_dim
        self.p_z_dim = p_z_dim
        self.p_x_dim = p_x_dim

        self.mode = mode
        if mode == 'gauss:':
            k = 1
        self.target_dim = k * x_dim

        # Feature extraction and transformation.
        self.X_transform = Sequential(
            [
                Dense(x2s_dim, activation='relu', name='l1'),
                Dense(x2s_dim, activation='relu', name='l2'),
            ],
            name='X_transform'
        )
        self.Z_transform = Sequential(
            [
                Dense(z2s_dim, activation='relu', name='l1'),
                Dense(z2s_dim, activation='relu', name='l2'),
            ],
            name='X_transform'
        )

        # Recurrence
        # [x2s_dim + z2s_dim] -> h_dim + [h_dim]
        self.rnn = GRULayer(h_dim)

        # Encoder
        # [x2s_dim + h_dim] -> q_z_dim -> z_dim
        #                              -> z_dim
        self.phi = Sequential(
            [
                Dense(q_z_dim, activation='relu', name='l1'),
                Dense(q_z_dim, activation='relu', name='l2'),
            ],
            name='phi'
        )
        self.phi_mu = Dense(z_dim, activation=None, name='phi_mu')
        self.phi_sig = Dense(z_dim, activation='softplus', name='phi_sig')

        # Prior
        # h_dim -> p_z_dim -> z_dim
        #                  -> z_dim
        self.prior = Sequential(
            [
                Dense(p_z_dim, activation='relu', name='l1'),
                Dense(p_z_dim, activation='relu', name='l2'),
            ],
            name='prior'
        )
        self.prior_mu = Dense(z_dim, activation=None, name='prior_mu')
        self.prior_sig = Dense(z_dim, activation='softplus', name='prior_sig')

        # Decoder
        # [z2s_dim, h_dim] -> p_z_dim -> target_dim
        #                             -> target_dim
        self.theta = Sequential(
            [
                Dense(p_z_dim, activation='relu', name='l1'),
                Dense(p_z_dim, activation='relu', name='l2'),
            ],
            name='theta'
        )
        self.theta_mu = Dense(self.target_dim, activation=None, name='theta_mu')
        self.theta_sig = Dense(self.target_dim, activation='softplus', name='theta_sig')
        self.coeff = Dense(k, activation='softmax', name='coeff')

        # Sampling
        self.sampling = Sampling()

        # Reconstruct
        self.reconsturct = Dense(x_dim, activation='tanh', name='reconstruction')

    def call(self, inputs, training=None, mask=None):
        batch = inputs.shape[0]
        timesteps = inputs.shape[1]

        X = self.X_transform(inputs)
        H1 = variable(zeros((batch, self.h_dim)))
        (s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_prime_temp, z) = self.inner_fn(X, H1)

        kl_temp = self.KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

        theta_input = concatenate([z_prime_temp, s_temp])
        theta = self.theta(theta_input)
        theta_mu = self.theta_mu(theta)
        theta_sig = self.theta_sig(theta)

        recon = self.Gaussian(inputs, theta_mu, theta_sig)

        recon_term = K.mean(recon, axis=-1)
        kl_term = K.mean(kl_temp, axis=-1)
        nll_upperbound = K.mean(recon_term + kl_term)

        self.add_loss(nll_upperbound)

        return theta_mu, theta_sig, z

    def inner_fn(self, inputs, initial_state):
        batch = inputs.shape[0]
        timesteps = inputs.shape[1]

        h_t = initial_state
        STATES = list()
        PHI_mu = list()
        PHI_sig = list()
        PRIOR_mu = list()
        PRIOR_sig = list()
        Z_prime = list()
        Z = list()
        for timestep in range(timesteps):
            x_t = inputs[:, timestep, :]

            phi = self.phi(concatenate([x_t, h_t]))
            phi_mu = self.phi_mu(phi)
            phi_sig = self.phi_sig(phi)

            prior = self.prior(h_t)
            prior_mu = self.prior_mu(prior)
            prior_sig = self.prior_sig(prior)

            z_t = self.sampling([phi_mu, phi_sig])
            zprime_t = self.Z_transform(z_t)

            _, [h_t] = self.rnn(inputs=concatenate([x_t, zprime_t]), states=[h_t])

            STATES.append(h_t)
            PHI_mu.append(phi_mu)
            PHI_sig.append(phi_sig)
            PRIOR_mu.append(prior_mu)
            PRIOR_sig.append(prior_sig)
            Z_prime.append(zprime_t)
            Z.append(z_t)

        STATES = K.stack(STATES, axis=1)
        phi_mu = K.stack(PHI_mu, axis=1)
        phi_sig = K.stack(PHI_sig, axis=1)
        prior_mu = K.stack(PRIOR_mu, axis=1)
        prior_sig = K.stack(PRIOR_sig, axis=1)
        z_prime = K.stack(Z_prime, axis=1)
        z = K.stack(Z, axis=1)

        return STATES, phi_mu, phi_sig, prior_mu, prior_sig, z_prime, z

    def KLGaussianGaussian(self, mu1, sig1, mu2, sig2, keep_dims=0):
        """
            Re-parameterized formula for KL
            between Gaussian predicted by encoder and Gaussian dist.

            Parameters
            ----------
            mu1  : FullyConnected (Linear)
            sig1 : FullyConnected (Softplus)
            mu2  : FullyConnected (Linear)
            sig2 : FullyConnected (Softplus)
            """
        if keep_dims:
            kl = 0.5 * (2 * K.log(sig2) - 2 * K.log(sig1) + (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2 - 1)
        else:
            kl = K.sum(0.5 * (2 * K.log(sig2) - 2 * K.log(sig1) + (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2 - 1),
                       axis=-1)

        return K.sum(kl, axis=-1)

    def Gaussian(self, y, mu, sig):
        """
        Gaussian negative log-likelihood

        Parameters
        ----------
        y   : TensorVariable
        mu  : FullyConnected (Linear)
        sig : FullyConnected (Softplus)
        """
        nll = 0.5 * K.sum(K.square(y - mu) / sig ** 2 + 2 * K.log(sig) + K.log(2 * np.pi), axis=-1)
        return nll

    def GMM(self, y, mu, sig, coeff):
        """
            Gaussian mixture model negative log-likelihood

            Parameters
            ----------
            y     : TensorVariable
            mu    : FullyConnected (Linear)
            sig   : FullyConnected (Softplus)
            coeff : FullyConnected (Softmax)
            """
        inner = -0.5 * K.sum(K.square(y - mu) / sig ** 2 + 2 * K.log(sig) + K.log(2 * np.pi), axis=-1)
        coeff_term = K.sum(coeff, axis=-1)
        nll = -self.logsumexp(K.log(coeff_term) + inner, axis=1)
        return K.sum(nll, axis=-1)

    def logsumexp(self, x, axis=None):
        x_max = K.max(x, axis=axis, keepdims=True)
        z = K.log(K.sum(K.exp(x - x_max), axis=axis, keepdims=True)) + x_max
        return z

    def sample(self, x_0, z):
        if len(x_0.shape) != 2:
            raise ValueError('x_0 dimension should be of size 2')
        batch = x_0.shape[0]
        if z.shape[0] != batch:
            raise ValueError('x_0 and z have incompatible batch dimension')
        timesteps = z.shape[1]

        reconstruction = list()
        xprime_t = self.X_transform(x_0)
        zprime = self.Z_transform(z)
        state_t = zeros((batch, self.h_dim))

        for timestep in range(timesteps):
            zprime_t = zprime[:, timestep, :]
            _, [state_t] = self.rnn(concatenate([xprime_t, zprime_t]), states=[state_t])

            theta_t = self.theta(concatenate([zprime_t, state_t]))
            theta_mu_t = self.theta_mu(theta_t)
            theta_sig_t = self.theta_sig(theta_t)

            reconstruction_t = self.sampling([theta_mu_t, theta_sig_t])
            xprime_t = self.X_transform(reconstruction_t)
            reconstruction.append(reconstruction_t)

        reconstruction = np.array(reconstruction)
        reconstruction = np.reshape(reconstruction, (batch, timesteps, -1))
        return reconstruction


if __name__ == '__main__':
    x_dim = 69
    x2s_dim = 150
    k = 1
    h_dim = 250
    z_dim = 69
    z2s_dim = 150
    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 150

    sequence = 100
    batch_size = 64

    x_seq = zeros((batch_size, sequence, x_dim))
    h = zeros((batch_size, h_dim))
    vrnn = Vrnn(x_dim=x_dim, x2s_dim=x2s_dim, k=k, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim, q_z_dim=q_z_dim,
                p_z_dim=p_z_dim, p_x_dim=p_x_dim)

    print("Testing rnn alone: ")
    t = 1
    x_t = zeros((batch_size, x2s_dim + z2s_dim))
    print("input x_t shape: ", x_t.shape)
    print("state shape: ", h.shape)
    output, [state] = vrnn.rnn(x_t, [h])
    print("output of rnn: ", output.shape)
    print("out state or rnn: ", state.shape)

    print("\n")
    print("testing the call methods: ")
    # vrnn.call_test1(x_seq)
    vrnn(x_seq)
    vrnn.sample_from_state(h, timesteps=sequence)
