from collections import OrderedDict
from numbers import Number
from copy import deepcopy
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .rl_algorithm import RLAlgorithm
from .utils import get_mirror_function

@tf.function(experimental_relax_shapes=True)
def td_targets(rewards, discounts, next_values):
    return rewards + discounts * next_values


@tf.function(experimental_relax_shapes=True)
def compute_Q_targets(next_Q_values,
                      next_log_pis,
                      rewards,
                      terminals,
                      discount,
                      entropy_scale,
                      reward_scale):
    next_values = next_Q_values - entropy_scale * next_log_pis
    terminals = tf.cast(terminals, next_values.dtype)

    Q_targets = td_targets(
        rewards=reward_scale * rewards,
        discounts=discount,
        next_values=(1.0 - terminals) * next_values)

    return Q_targets

class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            env,
            policy,
            Qs,
            pool,
            plotter=None,
            tf_summaries=False,
            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._env = env

        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(deepcopy(Q) for Q in Qs)
        self._update_target(tau=tf.constant(1.0))

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr
        self._alpha_lr = lr

        self._reward_scale = reward_scale

        self._target_entropy = (
            -np.prod(self._env.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        self._save_full_state = save_full_state

        observation_shape = self._env.active_observation_shape
        action_shape = self._env.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        self._log_alpha = tf.Variable(0.0)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)

        self._alpha_optimizer = tf.optimizers.Adam(
            self._alpha_lr, name='alpha_optimizer')

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, batch):

        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        entropy_scale = self._alpha
        reward_scale = self._reward_scale
        discount = self._discount

        next_actions, next_log_pis = self._policy.actions_and_log_probs(
            next_observations)

        tf.debugging.assert_shapes((
            (next_log_pis, ('B', 1)),
        ))
        tf.debugging.assert_shapes((
            (next_actions, ('B', 'nA')),
        ))

        next_Qs_values = tuple(
            Q([next_observations, next_actions]) for Q in self._Q_targets)

        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            next_log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, batch):
        """Update the Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._compute_Q_targets(batch)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))
        ))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q([observations, actions])

                Q_losses = 0.5 * (
                    tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))
                Q_loss = tf.nn.compute_average_loss(Q_losses)

            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self, batch):
        """Update the policy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations = batch['observations']

        with tf.GradientTape() as tape:
            actions, log_pis = self._policy.actions_and_log_probs(observations)

            Qs_log_targets = tuple(
                Q([observations, actions]) for Q in self._Qs)
            Q_log_targets = tf.reduce_min(Qs_log_targets, axis=0)
            policy_losses = self._alpha * log_pis - Q_log_targets
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        tf.debugging.assert_shapes((
            (actions, ('B', 'nA')),
            (log_pis, ('B', 1)),
            (policy_losses, ('B', 1)),
        ))

        policy_gradients = tape.gradient(
            policy_loss, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(
            policy_gradients, self._policy.trainable_variables))

        return policy_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_alpha(self, batch):
        if not isinstance(self._target_entropy, Number):
            return 0.0

        observations = batch['observations']

        actions, log_pis = self._policy.actions_and_log_probs(observations)

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
            # NOTE(hartikainen): It's important that we take the average here,
            # otherwise we end up effectively having `batch_size` times too
            # large learning rate.
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [self._log_alpha]))

        return alpha_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch):
        """Runs the update operations for policy, Q, and alpha."""
        self._update_critic(batch)
        self._update_actor(batch)
        self._update_alpha(batch)

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""
        return self._train(
            self._env,
            self._policy,
            self._pool,
            initial_exploration_policy=self._initial_exploration_policy
            )

    def _do_training(self, iteration, batch):

        self._do_updates(batch)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

    @tf.function(experimental_relax_shapes=True)
    def _get_diag_only(self,batch):
        Q_targets = self._compute_Q_targets(batch)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            Q_values = Q([observations, actions])
            Q_losses = 0.5 * (
                tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))

            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        actions, log_pis = self._policy.actions_and_log_probs(observations)

        Qs_log_targets = tuple(
            Q([observations, actions]) for Q in self._Qs)
        Q_log_targets = tf.reduce_min(Qs_log_targets, axis=0)
        policy_losses = self._alpha * log_pis - Q_log_targets

        tf.debugging.assert_shapes((
            (actions, ('B', 'nA')),
            (log_pis, ('B', 1)),
            (policy_losses, ('B', 1)),
        ))

        alpha_losses = -1.0 * (
            self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
        # NOTE(hartikainen): It's important that we take the average here,
        # otherwise we end up effectively having `batch_size` times too
        # large learning rate.

        return Qs_values, Qs_losses,policy_losses,alpha_losses,self._alpha

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        Qs_values, Qs_losses, policy_losses, alpha_losses,alpha=self._get_diag_only(batch)#,alpha
        Q_value, Q_loss,Q2_value, Q2_loss,  policy_losses, alpha_losses,alpha =Qs_values[0].numpy(), Qs_losses[0].numpy(),\
                                                           Qs_values[1].numpy(), Qs_losses[1].numpy(), \
                                                           policy_losses.numpy(), alpha_losses.numpy(),alpha.numpy()#,alpha,alpha.numpy()
        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_value),
            'Q-std': np.std(Q_value),
            'Q_loss': np.mean(Q_loss),
            'alpha_loss': np.mean(alpha_losses),
            'alpha': alpha,
            #'alpha': self._alpha.numpy(),

        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_alpha': self._alpha,

        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
