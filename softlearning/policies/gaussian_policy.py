"""GaussianPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from softlearning.models.feedforward import feedforward_model

from .base_policy import LatentSpacePolicy

SCALE_DIAG_MIN_MAX = (-20, 2)

class GaussianPolicy(LatentSpacePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 squash=True,
                 preprocessor=None,
                 name=None,
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        shift_and_log_scale_diag= self._shift_and_log_scale_diag_net(
            input_shapes=(conditions.shape[1:], ),
            output_size=output_shape[0] * 2,
        )(conditions)

        shift, actions_std = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=-1)
        )(shift_and_log_scale_diag)

        actions_std = tf.keras.layers.Lambda(
            lambda x: tf.math.softplus(x) + 1e-5)(actions_std)

        self.shift_and_scale_model= tf.keras.Model(self.condition_inputs, (shift, actions_std))
        self.actions_model = self.shift_and_scale_model


    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(GaussianPolicy, self).non_trainable_weights))

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):

        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            raw_actions=shifts
        else:

            raw_actions = shifts+tf.random.normal(shape=shifts.shape) * scales

        if self._squash:
            actions = tf.math.tanh(raw_actions)
        else:
            actions=raw_actions

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_pis(self, observations, actions):

        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            log_probs = tf.fill(
                tf.concat((tf.shape(shifts)[:-1], [1]), axis=0), np.inf)
        else:

            raw_actions = shifts
            raw_actions += tf.random.normal(shape=shifts.shape) * scales
            log_prob_u = tfp.distributions.MultivariateNormalDiag(
                loc=shifts,
                scale_diag=scales).log_prob(raw_actions)
            actions = tf.math.tanh(raw_actions)
            log_probs = log_prob_u - tf.reduce_sum(tf.math.log(1 - actions ** 2 + 1e-6),axis=1)
            if len(log_probs.shape)==1:
                log_probs=tf.expand_dims(log_probs,1)

        return log_probs


    @tf.function(experimental_relax_shapes=True)
    def actions_and_log_probs(self,observations):

        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            raw_actions =shifts
        else:
            raw_actions =shifts+ tf.random.normal(shape=shifts.shape) * scales

        if self._squash:
            actions = tf.math.tanh(raw_actions)
        else:
            actions =raw_actions


        if self._deterministic:
            log_probs = tf.fill(
                tf.concat((tf.shape(shifts)[:-1], [1]), axis=0), np.inf)
        else:
            log_prob_u = tfp.distributions.MultivariateNormalDiag(
                loc=shifts,
                scale_diag=scales).log_prob(raw_actions)

            log_probs = log_prob_u - tf.reduce_sum(tf.math.log(1 - actions ** 2 + 1e-6),axis=1)
            if len(log_probs.shape)==1:
                log_probs=tf.expand_dims(log_probs,1)

        return actions,log_probs

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        shifts_np, log_scale_diags_np = self.shift_and_scale_model(conditions)
        actions_np, log_pis_np = self.actions_and_log_probs(conditions)

        return OrderedDict({
            'shifts-mean': np.mean(shifts_np),
            'shifts-std': np.std(shifts_np),

            'log_scale_diags-mean': np.mean(log_scale_diags_np),
            'log_scale_diags-std': np.std(log_scale_diags_np),

            '-log-pis-mean': np.mean(-log_pis_np),
            '-log-pis-std': np.std(-log_pis_np),

            'actions-mean': np.mean(actions_np),
            'actions-std': np.std(actions_np),
        })

class FeedforwardGaussianPolicy(GaussianPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(FeedforwardGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        shift_and_log_scale_diag_net = feedforward_model(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shift_and_log_scale_diag_net
