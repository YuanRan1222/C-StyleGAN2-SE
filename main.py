from c_stylegan2 import *
from se import *
from utils import *


def elia_wind(mode):
    gan = C_StyleGAN2(batch_size=8,
                      c_dim=0,
                      data_size=(2, 768),
                      lr=1e-3,
                      multi=(1, 2, 4, 6, 8),
                      min_size=(2, 48),
                      mapping_layers=4,
                      nfilters=32,
                      name="elia_wind",
                      n_days=8,
                      mapping_dim=128,
                      z_dim=128,
                      display_network=False)

    se = SE(batch_size=8,
            epsilon=0.0004,
            gan=gan,
            lr=0.0005,
            lambda_rec=0.1,
            lambda_per=0.6,
            lambda_man=0.3,
            nfilters=16,
            name="elia_wind",
            n_days=8,
            w_dim=128)

    if mode == 0:  # Restart training c_stylegan2
        gan.train(steps=30000)
    elif mode == 1:  # Load existing model and continue training c_stylegan2
        gan.loadWeights()
        gan.train(steps=5000)
    elif mode == 2:  # Generate random samples using trained c_stylegan2
        gan.loadWeights()
        gan.generate()
    elif mode == 3:  # Restart training sequence encoder
        se.train(2)
    else:
        return


if __name__ == "__main__":
    allow_memory_growth()
    elia_wind(2)