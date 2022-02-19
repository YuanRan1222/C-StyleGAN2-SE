import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend, Model
from utils import plot_gif


class MDConv(layers.Layer):
    def __init__(self, filters: int, kernel_size: int, use_demodulate: bool, **kwargs):
        super(MDConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_demodulate = use_demodulate
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=2)]

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[0][1], self.kernel_size, input_shape[0][3], self.filters), initializer="glorot_uniform")

    def call(self, inputs, training=None):
        x = tf.transpose(inputs[0], [0, 3, 1, 2])
        s = backend.expand_dims(inputs[1], axis=1)
        s = backend.expand_dims(s, axis=1)
        s = backend.expand_dims(s, axis=-1)
        w = backend.expand_dims(self.kernel, axis=0) * (s + 1)

        if self.use_demodulate:
            w /= tf.math.sqrt(tf.math.reduce_sum(tf.math.square(w), axis=[1, 2, 3], keepdims=True) + 1e-8)

        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])
        w = tf.reshape(tf.transpose(w, [1, 2, 3, 0, 4]), [w.shape[1], w.shape[2], w.shape[3], -1])
        x = tf.nn.conv2d(x, w, strides=(1, 1), padding="SAME", data_format="NCHW")
        x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]])
        x = tf.transpose(x, [0, 2, 3, 1])

        return x


def Discriminator(nfilters: int, data_size: tuple, multi: tuple, c_dim=0):
    def block_with_c(x, cb, nfilters, use_pool):
        def concat(x, cb):
            cb = tf.keras.backend.tile(cb, [1, x.shape[1], x.shape[2], 1])
            return tf.concat([x, cb], axis=3)

        t = layers.Conv2D(filters=nfilters, kernel_size=(x.shape[1], 1), kernel_initializer="he_uniform")(concat(x, cb))
        x = layers.Conv2D(filters=nfilters, kernel_size=(x.shape[1], 3), padding="same", kernel_initializer="he_uniform")(concat(x, cb))
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters=nfilters, kernel_size=(x.shape[1], 3), padding="same", kernel_initializer="he_uniform")(concat(x, cb))
        x = layers.LeakyReLU(0.2)(x)
        x = layers.add([x, t])
        if use_pool:
            x = layers.AveragePooling2D(pool_size=(1, 2))(x)
        return x

    def block(x, nfilters, use_pool):
        t = layers.Conv2D(filters=nfilters, kernel_size=(x.shape[1], 1), kernel_initializer="he_uniform")(x)
        x = layers.Conv2D(filters=nfilters, kernel_size=(x.shape[1], 3), padding="same", kernel_initializer="he_uniform")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters=nfilters, kernel_size=(x.shape[1], 3), padding="same", kernel_initializer="he_uniform")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.add([x, t])
        if use_pool:
            x = layers.AveragePooling2D(pool_size=(1, 2))(x)
        return x

    x = layers.Input(shape=[data_size[0], data_size[1], 1])

    if c_dim:
        l, c = x, layers.Input(shape=(c_dim,))
        cb = layers.Reshape([1, 1, c.shape[1]])(c)
        for i in range(len(multi) - 1):
            l = block_with_c(l, cb, nfilters * multi[i], use_pool=True)
        l = block_with_c(l, cb, nfilters * multi[-1], use_pool=False)
        l = layers.Flatten()(l)
        l = layers.Dense(1, kernel_initializer="he_uniform")(tf.concat([l, c], axis=1))
        return Model(inputs=[x, c], outputs=l)
    else:
        l = x
        for i in range(len(multi) - 1):
            l = block(l, nfilters * multi[i], use_pool=True)
        l = block(l, nfilters * multi[-1], use_pool=False)
        l = layers.Flatten()(l)
        l = layers.Dense(1, kernel_initializer="he_uniform")(l)
        return Model(inputs=x, outputs=l)


def G_map(z_dim: int, mapping_dim: int, mapping_layers: int, c_dim=0):
    z = layers.Input(shape=(z_dim,))
    if c_dim:
        c = layers.Input(shape=(c_dim,))
        l = tf.concat([z, c], axis=1)
        for i in range(mapping_layers):
            l = layers.Dense(mapping_dim, activation=tf.nn.leaky_relu)(tf.concat([l, c], axis=1))
        return Model(inputs=[z, c], outputs=l)
    else:
        l = z
        for i in range(mapping_layers):
            l = layers.Dense(mapping_dim, activation=tf.nn.leaky_relu)(l)
        return Model(inputs=z, outputs=l)


def G_synthesis(data_size: tuple, nfilters: int, multi: tuple, min_size: tuple, mapping_dim: int):
    def toR(inputs, style):
        upsample = data_size[1] // inputs.shape[2]
        x = MDConv(filters=1, kernel_size=1, use_demodulate=False)([inputs, style])
        x = layers.UpSampling2D(size=(1, upsample), data_format="channels_last", interpolation="bilinear")(x)
        return x

    def block(inputs, style, nfilters, use_demodulate=True, use_up=True):
        x = inputs
        if use_up:
            x = layers.UpSampling2D(size=(1, 2), data_format="channels_last", interpolation="bilinear")(x)

        s = layers.Dense(inputs.shape[3], kernel_initializer="he_uniform")(style)
        x = MDConv(filters=nfilters, kernel_size=3, use_demodulate=use_demodulate)([x, s])
        x = layers.LeakyReLU(alpha=0.2)(x)
        s = layers.Dense(nfilters, kernel_initializer="he_uniform")(style)
        x = MDConv(filters=nfilters, kernel_size=3, use_demodulate=use_demodulate)([x, s])
        x = layers.LeakyReLU(alpha=0.2)(x)
        s = layers.Dense(nfilters, kernel_initializer="he_uniform")(style)

        return x, toR(x, s)

    style_input = []
    for i in range(len(multi) - 1):
        style_input.append(layers.Input([mapping_dim]))

    R = []
    x = layers.Lambda(lambda x: x[:, :1] * 0 + 1)(style_input[0])
    x = layers.Dense(data_size[0] * min_size[1] * nfilters, activation="relu", kernel_initializer="random_normal")(x)
    x = layers.Reshape([data_size[0], min_size[1], nfilters])(x)
    x, r = block(x, style_input[0], nfilters * multi[0], use_up=False)
    R.append(r)

    for i in range(len(multi) - 1):
        x, r = block(x, style_input[i], nfilters * multi[1 + i])
        R.append(r)

    x = tf.reduce_sum(R, axis=0)

    return Model(inputs=style_input, outputs=x)


class C_StyleGAN2:
    def __init__(self,
                 batch_size,
                 c_dim,
                 data_size,
                 lr,
                 multi,
                 min_size,
                 mapping_dim,
                 mapping_layers,
                 nfilters,
                 name,
                 n_days,
                 z_dim,
                 display_network=True):

        self.batch_size = batch_size
        self.c_dim = c_dim
        self.data_size = data_size
        self.lr = lr
        self.multi = multi
        self.min_size = min_size
        self.mapping_dim = mapping_dim
        self.mapping_layers = mapping_layers
        self.nfilters = nfilters
        self.name = name
        self.n_days = n_days
        self.z_dim = z_dim

        self.Gm = G_map(z_dim=z_dim, mapping_dim=mapping_dim, mapping_layers=mapping_layers, c_dim=c_dim)
        self.Gs = G_synthesis(data_size=data_size, nfilters=nfilters, multi=multi[::-1], min_size=min_size, mapping_dim=mapping_dim)
        self.D = Discriminator(nfilters=nfilters, data_size=data_size, multi=multi, c_dim=c_dim)
        self.pl_mean = 0.50
        self.pl_ema = 0.99
        self.gen_opt = optimizers.Adam(lr=lr)
        self.dis_opt = optimizers.Adam(lr=lr)

        self.create_generator()
        self.load_dataset()

        if display_network:
            self.Gm.summary()
            self.Gs.summary()
            self.D.summary()

    def create_generator(self):
        inputs, styles = [], []
        for i in range(len(self.multi) - 1):
            inputs.append([layers.Input(shape=(self.z_dim,), dtype=tf.float32), layers.Input(shape=(self.c_dim,), dtype=tf.float32)])
            styles.append(self.Gm(inputs[-1]))
        self.G = Model(inputs=inputs, outputs=self.Gs(styles))

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

        x, c = [], []
        for r in range(power_data.shape[0] - self.n_days):
            a = power_data[r:r + self.n_days]
            b = forecast_data[r:r + self.n_days]
            x.append(np.concatenate([a.reshape(1, -1, 1), b.reshape(1, -1, 1)], axis=0))
            c.append(meteo_data[r + self.n_days - 1].reshape(-1))
        x, c = np.array(x), np.array(c)
        self.dataset = tf.data.Dataset.from_tensor_slices((x, c)).shuffle(1000).batch(self.batch_size, drop_remainder=True).repeat()

    @tf.function
    def train_step(self, x, c, z, pl_reg):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            w = []
            for i in range(len(self.multi) - 1):
                w.append(self.Gm([z[i], c]))

            gen_output = self.Gs(w)

            dis_real = self.D([x, c])
            dis_fake = self.D([gen_output, c])

            gen_loss = tf.reduce_mean(dis_fake)
            dis_loss = tf.reduce_mean(tf.nn.relu(1 + dis_real) + tf.nn.relu(1 - dis_fake))

            grad = backend.square(backend.gradients(dis_real, x)[0])
            r1_loss = tf.reduce_mean(backend.sum(grad, axis=np.arange(1, len(grad.shape))))
            r1_loss = tf.cast(r1_loss, dtype=tf.float32)
            dis_loss += 2. * r1_loss

            pl_len = 0
            if pl_reg:
                w2 = []
                for i in range(len(self.multi) - 1):
                    std = 0.1 / (backend.std(w[i], axis=0, keepdims=True) + 1e-8)
                    w2.append(w[i] + backend.random_normal(tf.shape(w[i])) / (std + 1e-8))

                pl_output = self.Gs(w2)
                pl_len = tf.reduce_mean(backend.square(pl_output - gen_output), axis=[1, 2, 3])
                if self.pl_mean > 0:
                    pl_loss = tf.reduce_mean(backend.square(pl_len - self.pl_mean))
                    gen_loss += 5. * pl_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        gradients_of_discriminator = dis_tape.gradient(dis_loss, self.D.trainable_variables)

        self.gen_opt.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        self.dis_opt.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

        return gen_loss, dis_loss, pl_len

    def train(self, steps):
        for step, data in enumerate(self.dataset):
            x, c = data
            if np.random.rand() < 0.5:
                r = np.random.normal(size=(self.batch_size, self.mapping_dim))
                z = [r] * (len(self.multi) - 1)
            else:
                r1 = np.random.normal(size=(self.batch_size, self.mapping_dim))
                r2 = np.random.normal(size=(self.batch_size, self.mapping_dim))
                s = np.random.randint(1, len(self.multi) - 2)
                z = [r1] * s + [r2] * (len(self.multi) - 1 - s)

            g, d, pl = self.train_step(x, c, z, step % 20 == 0)
            g, d, pl = np.mean(g), np.mean(d), np.mean(pl)
            if step % 20 == 0:
                self.pl_mean = self.pl_ema * self.pl_mean + (1 - self.pl_ema) * pl

            if step % 100 == 0:
                print("Step: {:<6d} \t G: {:<.3f} \t D: {:<.3f}".format(step, g, d))

            if step == steps:
                break

        self.saveWeights()

    def generate(self, nums=100, z=None, c=None, display=True):
        if z is None:
            z = np.random.normal(size=(nums, self.mapping_dim))
        if c is None:
            meteo_data = np.array(pd.read_csv("{}/{}_meteorology.csv".format(self.name, self.name), header=None, index_col=0))
            c = meteo_data[np.random.choice(meteo_data.shape[0], size=nums, replace=False)]

        w = []
        for i in range(len(self.multi) - 1):
            w.append(self.Gm([z, c]))

        data = np.array(self.Gs(w)[:, :, :, 0])

        if display is True:
            plot_gif(data=data,
                     xlim=[0, self.data_size[1]],
                     ylim=[0, 1],
                     xticks_num=self.n_days + 1,
                     yticks_num=6,
                     xylabel=["Time (5min)", "Normalized Power (MW)"],
                     legend=["Generated Power", "Generated Forecast"],
                     saved_path="{}/generate.gif".format(self.name, self.name))
        return data

    def saveWeights(self):
        self.D.save_weights("{}/D.h5".format(self.name))
        self.Gs.save_weights("{}/Gs.h5".format(self.name))
        self.Gm.save_weights("{}/Gm.h5".format(self.name))

    def loadWeights(self):
        self.D.load_weights("{}/D.h5".format(self.name))
        self.Gs.load_weights("{}/Gs.h5".format(self.name))
        self.Gm.load_weights("{}/Gm.h5".format(self.name))