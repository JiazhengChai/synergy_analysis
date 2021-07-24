from copy import deepcopy
import numpy as np
import tensorflow as tf
def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm

def create_TD3_algorithm(variant, *args, **kwargs):
    from .TD3 import TD3

    algorithm = TD3(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,

    'TD3':create_TD3_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm

def get_mirror_function(indices):

    negation_obs_indices = indices[0]
    right_obs_indices = indices[1]
    left_obs_indices = indices[2]
    negation_action_indices = indices[3]
    right_action_indices = indices[4]
    left_action_indices = indices[5]

    #@tf.function(experimental_relax_shapes=True)
    def mirror_function(batch,np_array=True):#s, next_s, actions, rewards,terminals
        obs_batch = batch["observations"]
        next_obs_batch = batch["next_observations"]
        actions_batch = batch["actions"]
        reward_batch = batch["rewards"]
        done_batch = batch["terminals"]

        # Only observation and action needs to be mirrored
        if np_array:
            obs_clone = np.copy(obs_batch)
            next_obs_clone = np.copy(next_obs_batch)
            actions_clone = np.copy(actions_batch)
            def swap_lr(t, r, l):
                t[:, np.concatenate((r, l))] = t[:, np.concatenate((l, r))]
            obs_clone[:, negation_obs_indices] *= -1
            swap_lr(obs_clone, right_obs_indices, left_obs_indices)

            next_obs_clone[:, negation_obs_indices] *= -1
            swap_lr(next_obs_clone, right_obs_indices, left_obs_indices)

            actions_clone[:, negation_action_indices] *= -1
            swap_lr(actions_clone, right_action_indices, left_action_indices)
        else:
            obs_clone = tf.identity(obs_batch)
            next_obs_clone = tf.identity(next_obs_batch)
            actions_clone = tf.identity(actions_batch)

            def negation(tensor, neg_indices):
                updates = tf.gather(tensor, neg_indices, batch_dims=-1) * -1
                updates = tf.keras.backend.flatten(updates)
                my_indices = []
                for i in range(tensor.shape[0]):
                    for ni in neg_indices:
                        tmp = [i, ni]
                        my_indices.append(tmp)
                my_indices = tf.constant(my_indices)
                tensor=tf.tensor_scatter_nd_update(tensor, my_indices, updates)
                return tensor

            def lr_swap(tensor, l_indices,r_indices):
                updates = tf.gather(tensor, l_indices + r_indices, batch_dims=-1)
                updates = tf.keras.backend.flatten(updates)
                my_indices = []
                for i in range(tensor.shape[0]):
                    for ni in r_indices + l_indices:
                        tmp = [i, ni]
                        my_indices.append(tmp)
                my_indices = tf.constant(my_indices)
                tensor=tf.tensor_scatter_nd_update(tensor, my_indices, updates)
                return tensor

            obs_clone=negation(obs_clone,negation_obs_indices)
            obs_clone=lr_swap(obs_clone,left_obs_indices,right_obs_indices)

            next_obs_clone = negation(next_obs_clone, negation_obs_indices)
            next_obs_clone = lr_swap(next_obs_clone, left_obs_indices, right_obs_indices)

            if(len(negation_action_indices)!=0):
                actions_clone=negation(actions_clone,negation_action_indices)
            actions_clone=lr_swap(actions_clone,left_action_indices,right_action_indices)

        return {
            "observations":obs_clone,
            "next_observations":next_obs_clone,
            "actions":actions_clone,
            "rewards":reward_batch,
            "terminals":done_batch
        }

    return mirror_function
