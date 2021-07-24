import tempfile

import tensorflow as tf


class PicklableKerasModel(tf.keras.Model):
    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tf.keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}

        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()

            loaded_model = tf.keras.models.load_model(
                fd.name, custom_objects={
                    self.__class__.__name__: self.__class__})

        self.__dict__.update(loaded_model.__dict__.copy())

    @classmethod
    def from_config(cls, *args, custom_objects=None, **kwargs):
        custom_objects = custom_objects or {}
        custom_objects[cls.__name__] = cls
        custom_objects['tf'] = tf
        return super(PicklableKerasModel, cls).from_config(
            *args, custom_objects=custom_objects, **kwargs)


@tf.keras.utils.register_keras_serializable(package='Custom', name='myNorm')
class NormRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, lam=0., lam2=0.):
        self.lam = lam
        self.lam2 = lam2
    def __call__(self, x):
        assert len(x.shape) == 2

        norm_reg = tf.abs(tf.reduce_sum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x),axis=-1)) - 1,axis=0))

        mat=tf.matmul(x,x,transpose_b=True)
        upper_tri = (tf.linalg.band_part(tf.convert_to_tensor(mat, dtype=tf.float32), 0, -1) * (1 - tf.eye(x.shape[0])))
        matmul_sum = tf.reduce_sum(tf.abs(upper_tri))

        return self.lam * norm_reg + self.lam2 *matmul_sum

    def get_config(self):
        return {'lam': float(self.lam),'lam2': float(self.lam2)}