# Genetic Soft Actor Cricit
Discrete Soft Actor Critic algorithm with Prioritized Experience Replay Buffer

## EN: Run with Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeiSovik/GSAC/blob/master/sac.en.ipynb)

## RU: Запустить с Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeiSovik/GSAC/blob/master/sac.ru.ipynb)

# Notice

Original work Copyright (c) June 2020, Sergei Sovik <sergeisovik@yahoo.com>

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

The software is provided `as is` and the author disclaims all warranties with regard to this software including all implied warranties of merchantability and fitness. In no event shall the author be liable for any special, direct, indirect, or consequential damages or any damages whatsoever resulting from loss of use, data or profits, whether in an action of contract, negligence or other tortious action, arising out of or in connection with the use or performance of this software.

# Foreword
The algorithm had to be implemented on old Tensorflow functions with Eager disabled, due to the fact that new functions of Tensorflow 2.2 led to a large memory leak and 32 GB of my memory were consumed in literally one hour.

All names of variables and classes are given based on programming experience for more than 25 years, possibly unusual, but intuitive for readers and not a generally accepted standard.

This article was written with the aim of expanding the circle of users with the `Soft Actor Critic` algorithm, since at the time of writing this article, it is the best, and all existing articles are written in a language incomprehensible to many programmers.

For those who absolutely do not understand what a `Computional Graph` is and does not want to go into details. This is a model describing the relationship between all calculations, including determining their order of execution. Each operation is called a `Node`. A `Computional Graph` is somewhat reminiscent of a block diagram with many possible inputs and outputs. Thus, requesting to calculate the result of the `Node` from the neural network engine, all dependencies are calculated, and if necessary, input data is requested.

# Algorithm `Genetic Soft Actor Critic`
The algorithm is implemented as a single graph, which allows to reduce the amount of data exchange with the GPU, and speed up the learning process.

The algorithm consists of four main blocks:
- Block `Neural Network`
- Block `Player`
- Block `Genetic Replay Buffer`
- Block `Trainer`

Each of which can work in parallel-serial.

Block `Neural network`
A neural network consists of several independently trained blocks:
- Two subnet clones `Trainer Actor` and `Target Actor`
- Two duplicate subnets `Trainer Critic`
- Two subnets `Target Critic`
- Coefficient `Alpha Regulator`

### Two subnet clones `Trainer Actor` and `Target Actor`
The `Target Actor` is used exclusively for the ability to parallelize the training and filling in the `Replay Buffer` with new data and is a complete copy of the `Trainer Actor` neural network.

### Two duplicate subnets `Trainer Critic`
Necessary to minimize errors.

### Two subnets `Target Critic`
Used for smooth learning using the moving average method.

### Coefficient `Alpha Regulator`
Performs the role of micro-adjustment of the learning process, to increase accuracy.

## Block `Player`
There is a certain environment in which it is necessary to carry out certain actions to achieve the goal. To simplify understanding, let's call the environment `Game`. The task of the `Player` is to collect observation data from the `Game`, to perform actions, and to receive a `Reward` from the `Game` or to independently make a `Rating` of these actions. Every completed action is a step. In one step, we have the following data set: `Previous Observation`, `Current Observation`, `Completed Action`, `Reward` or `Rating`, `End status`. The decision about which action to take is made by the `Target Actor` network based on the data of the `Previous Observation`. If the decision leads to a situation that can be considered the end, the `Player` completes and resets the` Game`.

There is two types of ratings:
- Rate rewards for every step
- Rate of the entire episode

Each step is stored in the `Replay Buffer` for further training and is called the `Trajectory`

### Rate rewards for every step
The `Player` takes an action and immediately writes to the `Replay Buffer` the following indicators: `Previous Observation`, `Current Observation`, `Completed Action`, `Reward`. Then `Rating` produced by `Trainer`.

### Rate of the entire episode
The `Player` takes action and stores to the `Temporary Buffer`. At the end of the episode, it calculates the `Rating` at each step and then stores to the `Replay Buffer` of the entire episode with the following indicators: `Previous Observation`, `Current Observation`, `Completed Action`, ` Rating`.

## Block `Genetic Replay Buffer`
It is a cyclic `Repeat Buffer`, which, when overflowed, starts overwriting older data with newer ones. It also includes the `Tree-based buffer of the sum` and the` Tree-based buffer of the maximum` used to calculate the priority of each step stored in the `Replay Buffer`. The tree-based buffers in a pair have the similarity with the genetic algorithm when selecting data from the `Replay Buffer`, which can greatly accelerate the learning process, and also reduces the likelihood of knocking down or freezing of the trained model in poor condition. A poor condition can be the result of the neural network getting used to poor results.

## Block `Trainer`
The main brain center of the algorithm that controls all the other blocks.

The training cycle for each step of the `Player`:
1. Select a batch of `Trajectories` from the `Genetic Replay Buffer`, taking into account the priorities.
2. Train two `Trainer Critics` independently.
3. Update the `Target Critics` using the `Moving Average` method.
4. Train `Trainer Actor` and `Alpha Regulator`.
5. Update the `Target Actor`.
6. Update priorities in the `Genetic Replay Buffer` for processed steps from a batch.

The learning process is standard: forward distribution, loss calculation, gradient calculation, back propagation.
