# Conditional Style-based Generative Adversarial Networks for Renewable Scenario Generation
This repository contains source code necessary to reproduce the results presented in the following paper:

Conditional Style-based Generative Adversarial Networks for Renewable Scenario Generation, Under review in *IEEE Transactions on Power Systems*.

Authors: Ran Yuan, Bo Wang, Yeqi Sun, Xuanning Song, Junzo Watada.

## Introduction

Day-ahead scenario generation of renewable power plays an important role in short-term power system operations due to considerable output uncertainty included. Most of Generative Adversarial Networks (GAN) based scenario generation approaches directly adopt auxiliary models or gradient descent optimization as their learning principles of data-mining and feature-extracting, which may suffer from parameters non-convergence, performance instability, and limited generalization capacity, leading to inadequate identification of spatial-temporal distribution and diurnal pattern correlation easily. Therefore, a deep renewable scenario generation model using conditional style-based generative adversarial networks followed by a sequence encoder network (nominated as C-StyleGAN2-SE), was developed to generate day-ahead scenarios directly from historical data through different-level scenario style controlling and mixing. The integration of meteorological information serving as conditions enables the model to capture the complex diurnal pattern and seasonality difference of renewable power.

## Requirements

- 64-bit Python 3.7 installation.
- Tensorflow 2.2.0.
- No less than 16G RAM.
- One or more high-end NVIDIA GPUs is highly recommended to accelerate training process.

## Randomly Generated Sample
![generate](/assets/generate.gif)

## Scenario Style Mixing
![StyleMix](/assets/StyleMix.png)

## Contact

For more information about code and methods, please feel free to contact Ran Yuan: yuanran1222@163.com

## Notice

There is a Github limit on the size of file (no more than 25M). Some codes and the nrel_wind dataset were not uploaded. Additionally, only the saved network models (.h5 files) trained on elia_wind dataset were uploaded. If you have any inquires on that particular dataset, please feel free to ask me.
