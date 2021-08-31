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
        self.GRU4 = GRUCell(hdim, name='rnn4')

    def call(self, inputs, states):
        [state1, state2, state3, state4] = states
        o1, [s1] = self.GRU1(inputs, [state1])
        o2, [s2] = self.GRU1(inputs, [state2])
        o3, [s3] = self.GRU1(inputs, [state3])
        outputs, [s4] = self.GRU2(o1, [state4])
        state_prime = [s1, s2, s3, s4]
        return outputs, state_prime

    def get_initial_state(self, inputs):
        h1 = self.GRU1.get_initial_state(inputs=inputs)
        h2 = self.GRU2.get_initial_state(inputs=inputs)
        h3 = self.GRU3.get_initial_state(inputs=inputs)
        return [h1, h2, h3]


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

        # Save state
        self.saved_state = None
        self.saved_z = None
        self.saved_zprime = None

    def call(self, inputs, training=None, mask=None):
        batch = inputs.shape[0]
        timesteps = inputs.shape[1]

        X = self.X_transform(inputs)
        H1 = variable(zeros((batch, self.h_dim)))
        H2 = variable(zeros((batch, self.h_dim)))
        H3 = variable(zeros((batch, self.h_dim)))
        H4 = variable(zeros((batch, self.h_dim)))
        H = [H1, H2, H3, H4]
        (s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_prime_temp, z) = self.inner_fn(X, H)

        self.saved_state = s_temp
        self.saved_zprime = z_prime_temp
        self.saved_z = z

        kl_temp = self.KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

        theta_input = concatenate([z_prime_temp, s_temp])
        theta = self.theta(theta_input)
        theta_mu = self.theta_mu(theta)
        theta_sig = self.theta_sig(theta)

        if self.mode == 'gauss':
            recon = self.Gaussian(inputs, theta_mu, theta_sig)
        else:
            coeff = self.coeff(theta)
            recon = self.GMM(inputs, theta_mu, theta_sig)

        recon_term = K.mean(recon, axis=-1)
        kl_term = K.mean(kl_temp, axis=-1)
        nll_upperbound = K.mean(recon_term + kl_term)

        self.add_loss(nll_upperbound)

        return theta_mu, theta_sig, z

    def inner_fn(self, inputs, initial_states):
        timesteps = inputs.shape[1]

        h_t = initial_states
        STATES = list()
        PHI_mu = list()
        PHI_sig = list()
        PRIOR_mu = list()
        PRIOR_sig = list()
        Z_prime = list()
        Z = list()
        for timestep in range(timesteps):
            x_t = inputs[:, timestep, :]

            phi = self.phi(concatenate([x_t, h_t[-1]]))
            phi_mu = self.phi_mu(phi)
            phi_sig = self.phi_sig(phi)

            prior = self.prior(h_t[-1])
            prior_mu = self.prior_mu(prior)
            prior_sig = self.prior_sig(prior)

            z_t = self.sampling([phi_mu, phi_sig])
            zprime_t = self.Z_transform(z_t)

            _, h_t = self.rnn(inputs=concatenate([x_t, zprime_t]), states=h_t)

            STATES.append(h_t[-1])
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

    def sample_from_state(self, initial_state, timesteps):
        """
        Only developed for gauss mode...
        :param initial_state:
        :param timesteps:
        :return:
        """
        X = list()
        state_t = initial_state
        for timestep in range(timesteps):
            prior_t = self.prior(state_t)
            prior_mu_t = self.prior_mu(prior_t)
            prior_sig_t = self.prior_sig(prior_t)

            z_t = self.sampling([prior_mu_t, prior_sig_t])
            zprime_t = self.Z_transform(z_t)

            theta_t = self.theta(concatenate([zprime_t, state_t]))
            theta_mu_t = self.theta_mu(theta_t)
            theta_sig_t = self.theta_sig(theta_t)

            x_t = self.sampling([theta_mu_t, theta_sig_t])
            xprime_t = self.X_transform(x_t)

            _, [state_t] = self.rnn(concatenate([xprime_t, zprime_t]), [state_t])

            X.append(x_t)
        X = np.array(X)
        return X

    def sample(self, x_0, z, return_state=False):
        if len(x_0.shape) != 2:
            raise ValueError('x_0 dimension should be of size 2')
        batch = x_0.shape[0]
        if z.shape[0] != batch:
            raise ValueError('x_0 and z have incompatible batch dimension')
        timesteps = z.shape[1]

        state_list = list()
        zprime_list = list()

        reconstruction = list()
        xprime_t = self.X_transform(x_0)
        zprime = self.Z_transform(z)
        H1 = variable(zeros((batch, self.h_dim)))
        H2 = variable(zeros((batch, self.h_dim)))
        H3 = variable(zeros((batch, self.h_dim)))
        H4 = variable(zeros((batch, self.h_dim)))
        state_t = [H1, H2, H3, H4]

        for timestep in range(timesteps):
            zprime_t = zprime[:, timestep, :]
            _, state_t = self.rnn(concatenate([xprime_t, zprime_t]), states=state_t)

            theta_t = self.theta(concatenate([zprime_t, state_t[-1]]))
            theta_mu_t = self.theta_mu(theta_t)
            theta_sig_t = self.theta_sig(theta_t)

            reconstruction_t = self.sampling([theta_mu_t, theta_sig_t])

            xprime_t = self.X_transform(reconstruction_t)
            reconstruction.append(reconstruction_t)

            state_list.append(state_t[-1])
            zprime_list.append(zprime_t)

        reconstruction = np.array(reconstruction)
        reconstruction = np.reshape(reconstruction, (batch, timesteps, -1))
        if return_state:
            return reconstruction, state_list, zprime_list
        else:
            return reconstruction

    def comparison(self, to_compare, with_what):
        if with_what == 'state':
            to_compare = np.squeeze(to_compare)
            self.saved_state = np.squeeze(self.saved_state)
            comparison = np.mean(self.saved_state - to_compare, axis=-1)
        if with_what == 'z':
            to_compare = np.squeeze(to_compare)
            self.saved_z = np.squeeze(self.saved_z)
            comparison = np.mean(self.saved_z - to_compare, axis=-1)
        if with_what == 'zprime':
            to_compare = np.squeeze(to_compare)
            self.saved_zprime = np.squeeze(self.saved_zprime)
            comparison = np.mean(self.saved_zprime - to_compare, axis=-1)
        return comparison

    def call_test1(self, inputs, training=None, mask=None):
        batch_size = inputs.shape[0]
        sequence = inputs.shape[1]

        print("inputs shape = ", inputs.shape)
        print('\n')

        X = self.X_transform(inputs)
        print("first transformation: ", X.shape)
        H = variable(zeros((batch_size, self.h_dim)))
        print('\n')
        print("Initial State shape: ", H.shape)
        (s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_1_temp) = self.inner_fn(X, H)
        print('\n')
        print("After inner_fn :")
        print("s_temp: ", s_temp.shape)
        print("phi_mu_temp: ", phi_mu_temp.shape)
        print("phi_sig_temp: ", phi_sig_temp.shape)
        print("prior_mu_temp: ", prior_mu_temp.shape)
        print("prior_sig_temp: ", prior_sig_temp.shape)
        print("z_1_temp: ", z_1_temp.shape)

        kl_temp = self.KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)
        print("kl term: ", kl_temp.shape)

        print('\n')
        print("Last transformations : ")
        theta = self.theta(concatenate([z_1_temp, s_temp]))
        theta_mu = self.theta_mu(theta)
        theta_sig = self.theta_sig(theta)
        coeff = self.coeff(theta)
        print("theta: ", theta.shape)
        print("theta_mu: ", theta_mu.shape)
        print("theta_sig: ", theta_sig.shape)
        print("coeff : ", coeff.shape)

        if self.mode == 'gauss':
            recon = self.Gaussian(inputs, theta_mu, theta_sig)
        else:
            coeff = self.coeff(theta)
            recon = self.GMM(inputs, theta_mu, theta_sig)
        print("recon term: ", recon.shape)

        recon_term = K.mean(recon, axis=-1)
        kl_term = K.mean(kl_temp, axis=-1)
        print("kl term: ", kl_temp.shape)
        print("recon term: ", recon.shape)
        nll_upperbound = recon_term + kl_term
        nll_upperbound = K.mean(nll_upperbound)
        print(nll_upperbound.shape)
        # self.add_loss(nll_upperbound)


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
