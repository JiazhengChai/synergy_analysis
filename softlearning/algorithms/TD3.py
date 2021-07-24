from collections import OrderedDict
from numbers import Number
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .rl_algorithm import RLAlgorithm

@tf.function(experimental_relax_shapes=True)
def td_target(reward, discount, next_value):
    return reward + discount * next_value

class TD3(RLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            Qs,
            pool,
            plotter=None,
            tf_summaries=False,

            lr=1e-3,
            reward_scale=1.0,
            discount=0.99,
            tau=5e-3,
            target_update_interval=2,
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
        """

        super(TD3, self).__init__(**kwargs)

        self._env = env
        self._policy = policy
        #self._policy_targets =tf.keras.models.clone_model(policy.deterministic_actions_model)
        self._policy_targets = deepcopy(self._policy)

        self._Qs = Qs
        self._Q_targets = tuple(deepcopy(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval

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

    @tf.function(experimental_relax_shapes=True)
    def _get_Q_target(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        reward_scale = self._reward_scale
        discount = self._discount

        next_actions = self._policy_targets.actions(next_observations)
        tf.debugging.assert_shapes((
            (next_actions, ('B', 'nA')),
        ))

        eps=tf.clip_by_value(tf.random.normal(shape=tf.shape(next_actions),stddev=0.2),-0.5,0.5)
        tf.debugging.assert_shapes((
            (eps, ('B', 'nA')),
        ))

        next_Qs_values = tuple(
            Q([next_observations, tf.clip_by_value(next_actions+eps,self._env.action_space.low,self._env.action_space.high)])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)

        tf.debugging.assert_shapes((
            (min_next_Q, ('B', 1)),
        ))

        next_value = min_next_Q

        Q_target = td_target(
            reward=reward_scale * rewards,
            discount=discount,
            next_value=(1 - tf.cast(terminals,tf.float32)) * next_value)

        tf.debugging.assert_shapes((
            (Q_target, ('B', 1)),
        ))

        return Q_target

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, batch):
        Q_target = tf.stop_gradient(self._get_Q_target(batch))

        tf.debugging.assert_shapes((
            (Q_target, ('B', 1)),
        ))

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_value = Q([observations, actions])

                Q_loss = 0.5 * (
                    tf.losses.MSE(y_true=Q_target, y_pred=Q_value))
                Q_loss = tf.nn.compute_average_loss(Q_loss)

            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_loss)
            Qs_values.append(Q_value)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self,batch):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations = batch['observations']
        with tf.GradientTape() as tape:
            actions = self._policy.actions(observations)
            min_Q_log_target= self._Qs[0]([observations, actions])

            policy_losses =  - min_Q_log_target
            policy_loss = tf.nn.compute_average_loss(policy_losses)

            tf.debugging.assert_shapes((
                (actions, ('B', 'nA')),
                (policy_losses, ('B', 1)),
            ))

        policy_gradients = tape.gradient(
            policy_loss, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(
            policy_gradients, self._policy.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        #tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

        for psource_weight, ptarget_weight in zip(
                self._policy.trainable_variables, self._policy_targets.trainable_variables):
            ptarget_weight.assign(
                tau * psource_weight + (1.0 - tau) * ptarget_weight
            )

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch):
        """Runs the update operations for policy, Q, and alpha."""
        self._update_critic(batch)
        self._update_actor(batch)

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
        Q_target = self._get_Q_target(batch)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        tf.debugging.assert_shapes((
            (Q_target, ('B', 1)), (rewards, ('B', 1))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            Q_values = Q([observations, actions])
            Q_losses = 0.5 * (
                tf.losses.MSE(y_true=Q_target, y_pred=Q_values))

            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        actions = self._policy.actions(observations)

        min_Q_log_target = self._Qs[0]([observations, actions])
        policy_losses = - min_Q_log_target

        tf.debugging.assert_shapes((
            (actions, ('B', 'nA')),
            (policy_losses, ('B', 1)),
        ))

        return Qs_values, Qs_losses,policy_losses

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):

        Qs_values, Qs_losses, policy_losses=self._get_diag_only(batch)
        Q_value, Q_loss, Q2_value, Q2_loss, policy_losses = Qs_values[0].numpy(), Qs_losses[0].numpy(), \
                                                                          Qs_values[1].numpy(), Qs_losses[1].numpy(), \
                                                                          policy_losses.numpy()

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_value),
            'Q-std': np.std(Q_value),
            'Q_loss': np.mean(Q_loss),

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
        }

        return saveables