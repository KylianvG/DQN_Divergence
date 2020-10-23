'''
The code in this file is largely copied from the DQN_lab jupyter notebook
provided as an assignment for the Reinforcement Learning course at the
University of Amsterdam, and modified where necessary.
'''

import torch
import torch.nn.functional as F

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.

    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    q_values = torch.gather(Q(states), 1, actions)
    return q_values

def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).

    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    q_next = Q(next_states)
    max_q, _ = torch.max(q_next, dim=1, keepdim=True)
    targets = rewards + discount_factor * torch.logical_not(dones) * max_q
    return targets

def clipped_loss(q_val, target, clip = 1.0):
    delta = torch.abs(q_val - target)
    if clip >= 0.0:
        # Since delta >= 0.0, we need to clamp only from 0.0 to the clip value. Clamping delta from 0.0 to clip is the 
        # same as clamping the error between -clip and clip. 
        quad = torch.clamp(delta, 0.0, clip)
        lin = delta - quad
        loss = (0.5 * torch.square(quad)) + (clip * lin)
    else:
        loss = (0.5 * torch.square(delta))
    return torch.sum(loss)

def train(Q, memory, optimizer, batch_size, discount_factor, clip):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    Q.train()

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean

    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = clipped_loss(q_val, target, clip) 

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())
