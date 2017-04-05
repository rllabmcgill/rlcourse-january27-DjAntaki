
import numpy as np
from utils import sample, onehot


def random_prob_array(size):
    q = np.random.uniform(0,1,size)
    return 1.0/np.sum(q) * q

def random_transition_array(nstates):
    a,b = np.random.choice(range(nstates),size=2,replace=False)
    p = np.random.uniform()
    return p*onehot(nstates,a) + (1-p)*onehot(nstates,b)

class RandomMDP:
    def __init__(self,nstates,nactions):
        self.num_actions = nactions
        self.num_states = nstates
        self.nstates = nstates
        self.transition_matrix = np.array([[random_transition_array(nstates) for a in range(self.num_actions)] for n in range(nstates)])
#        self.transition_matrix = np.array([[random_prob_array((nstates,)) for a in range(self.num_actions)] for n in range(nstates)])
        self.reward = np.random.uniform(low=-1.0,size=nstates*self.num_actions).reshape((nstates,self.num_actions))

    def reset(self):
        self.current_state = 0
        return 0

    def is_terminal(self):
        return self.current_state == self.nstates -1

    def step(self,action):

        transition_probs = self.transition_matrix[self.current_state][action]

        next_state = sample(transition_probs)
        reward = self.reward[self.current_state][action]

        self.current_state = next_state
        isterminal = False

        if self.is_terminal():
            isterminal = True
        #    reward += 100
        return self.current_state, reward, isterminal, {}

