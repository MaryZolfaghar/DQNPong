"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import numpy as np
import random

def remap_actions(actions):
    """
    re-map actions to include only three main actions
    0: Noop         (stay)
    1: Fire         (stay)
    2: Right        (up)
    3: Left         (down)
    4: Right-Fire   (up)
    5: Left-Fire    (down)
    """

    valid_actions = {'stay':0, 'up': 2, 'down': 3}
    actions[actions==1] = valid_actions['stay']
    actions[actions==4] = valid_actions['up']
    actions[actions==5] = valid_actions['down']
