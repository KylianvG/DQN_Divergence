# DQN_Divergence
Project studying the divergence of Deep Q-Learning Networks. This small study was conducted as part of the Reinforcement Learning (2020/2021) course at the University of Amsterdam.

We study the following potential measures to combat divergence in the CartPole-v1 environment:
* Experience replay
* Target networks
* Batch normalisation
* Error clipping

We find that target networks, batch normalisation and error clipping can be effective, while experience replay by itself appears to not have an effect on divergence.

Each countermeasure is experimented with in its own dedicated notebook.
