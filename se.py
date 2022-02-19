import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model


class SequenceEncoder(Model):
    def __init__(self, nfilters, w_dim):
        super(SequenceEncoder, self).__init__()
        self.nfilters = nfilters
        self.w_dim = w_dim

    def build(self, input_shape):
        x_shape, c_shape = input_shape[0], input_shape[1]
        self.fc1 = layers.Dense(units=c_shape[1] // x_shape[1])
        self.fc2 = layers.Dense(units=self.w_dim)
        self.convlstm1 = layers.ConvLSTM2D(filters=self.nfilters, kernel_size=(x_shape[2], 3), strides=(1, 2), padding="same", return_sequences=True)
        self.convlstm2 = layers.ConvLSTM2D(filters=self.nfilters * 2, kernel_size=(x_shape[2], 3), strides=(1, 2), padding="same", return_sequences=False)
        self.conv1 = layers.Conv2D(filters=c_shape[1] // x_shape[1], kernel_size=(1, 3), strides=(1, 2), padding="same")
        self.conv2 = layers.Conv2D(filters=c_shape[1] // x_shape[1], kernel_size=(1, 3), strides=(1, 1), padding="same")
        self.conv3 = layers.Conv2D(filters=self.nfilters * 2, kernel_size=(1, 3), strides=(1, 1), padding="same")
        self.conv4 = layers.Conv2D(filters=self.nfilters * 4, kernel_size=(1, 3), strides=(1, 1), padding="same")
        self.pool = layers.GlobalAveragePooling2D()

    def concat_fc_and_convlstm(self, f, cl):
        f = tf.tile(tf.reshape(f, shape=[f.shape[0], 1, 1, f.shape[1]]), [1, cl.shape[2], cl.shape[3], 1])
        cl = tf.reshape(tf.transpose(cl, [0, 2, 3, 1, 4]), shape=[cl.shape[0], cl.shape[2], cl.shape[3], cl.shape[1] * cl.shape[4]])
        return tf.concat([cl, f], axis=3)

    def call(self, inputs, training=None, mask=None):
        x, c = inputs
        f1 = self.fc1(c)
        cl1 = self.convlstm1(x)
        cl2 = self.convlstm2(cl1)
        c1 = self.conv1(self.concat_fc_and_convlstm(f1, cl1))
        c2 = self.conv2(c1)
        c3 = self.conv3(tf.concat([c1, cl2], axis=3))
        c4 = self.conv4(tf.concat([c2, c3], axis=3))
        f2 = self.fc2(self.pool(c4))
        return f2


class SE:
    def __init__(self,
                 batch_size,
                 epsilon,
                 gan,
                 lr,
                 lambda_rec,
                 lambda_per,
                 lambda_man,
                 nfilters,
                 name,
                 n_days,
                 w_dim):

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gan = gan
        self.lr = lr
        self.lambda_rec = lambda_rec
        self.lambda_per = lambda_per
        self.lambda_man = lambda_man
        self.nfilters = nfilters
        self.name = name
        self.n_days = n_days
        self.w_dim = w_dim

        self.opt = optimizers.Adam(lr=lr)

        self.encoder = SequenceEncoder(nfilters=nfilters, w_dim=w_dim)
        self.load_gan()
        self.load_dataset()

    def load_dataset(self):
        power_data = np.array(pd.read_csv("{}/{}_power.csv".format(self.name, self.name), header=None, index_col=0))
        forecast_data = np.array(pd.read_csv("{}/{}_forecast.csv".format(self.name, self.name), header=None, index_col=0))
        """
        Please note that when using the weather API of wunderground website to obtain meteorological data,
        you first need to register a personal account from https://www.wunderground.com/.
        The readers should replace with your downloaded meteorological data, and then use as:        
        meteo_data = np.array(pd.read_csv(...))
        """
        meteo_data = np.array(pd.read_csv("{}/{}_meteorology.csv".format(self.name, self.name), header=None, index_col=0))

        xs, c, x = [], [], []
        for r in range(power_data.shape[0] - self.n_days):
            a = power_data[r:r + self.n_days]
            b = forecast_data[r:r + self.n_days]
            xs.append(np.stack([np.concatenate([a[:self.n_days - 1], b[self.n_days - 1:]], axis=0), b], axis=1))
            c.append(meteo_data[r:r + self.n_days].reshape(-1))
            x.append(np.concatenate([a.reshape(1, -1, 1), b.reshape(1, -1, 1)], axis=0))
        xs, c, x = np.expand_dims(xs, axis=4), np.array(c), np.array(x)
        self.dataset = tf.data.Dataset.from_tensor_slices((xs, c, x)).shuffle(1000).batch(self.batch_size, drop_remainder=True).repeat()

    def load_gan(self):
        self.Gm = self.gan.Gm
        self.Gs = self.gan.Gs
        self.D = self.gan.D
        self.gan.loadWeights()
        self.Gm.trainable = False
        self.Gs.trainable = False
        self.D.trainable = False

    @tf.function
    def train_step(self, xs, c, x, step):
        with tf.GradientTape() as tape:
            w = [self.encoder([xs, c]) for _ in range(len(self.gan.multi) - 1)]
            xr = self.Gs(w)
            w_exp = []
            for _ in range(self.batch_size):
                w_exp.append(tf.reduce_mean(self.Gm([np.random.normal(size=(100, self.gan.mapping_dim)), c]), axis=0, keepdims=True))
            w_exp = tf.concat(w_exp, axis=0)

            loss_rec = tf.reduce_mean(tf.square(xr - x))
            loss_per = tf.reduce_mean(tf.abs(self.D([xr, c]) - self.D([x, c])))
            loss_man = tf.math.exp(-self.epsilon * step) * tf.reduce_mean(tf.square(w_exp - w[0]))

            encoder_loss = self.lambda_rec * loss_rec + self.lambda_per * loss_per + self.lambda_man * loss_man

        gradients = tape.gradient(encoder_loss, self.encoder.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        return encoder_loss

    def train(self, steps):
        for step, data in enumerate(self.dataset):
            xs, c, x = data
            e = self.train_step(xs, c, x, step)

            if step % 100 == 0:
                print("Step: {:<6d} \t SE: {:<.3f}".format(step, e))

            if step == steps:
                break

        self.saveWeights()

    def saveWeights(self):
        self.encoder.save_weights("{}/SE.h5".format(self.name))

    def loadWeights(self):
        self.encoder.load_weights("{}/SE.h5".format(self.name))