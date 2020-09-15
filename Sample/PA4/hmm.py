from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.key_list = list(self.state_dict.keys()) 
        self.val_list = list(self.state_dict.values())

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        for i in range(S):
            alpha[i][0]=self.pi[i]*self.B[i][self.obs_dict[Osequence[0]]]
        for j in range(1,L):
            for i in range(S):
                alpha[i][j]=self.B[i][self.obs_dict[Osequence[j]]]*np.sum(self.A[:,i]*alpha[:,j-1])
        ###################################################
        # Edit here
        ###################################################
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        beta[:,L-1]=np.ones((S))
        for j in reversed(range(L-1)):
            for i in range(S):
                beta[i][j]=np.sum(self.B[:,self.obs_dict[Osequence[j+1]]]*beta[:,j+1]*self.A[i])
        ###################################################
        # Edit here
        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        L = len(Osequence)
        a = self.forward(Osequence)
        prob = np.sum(a[:,L-1])
        ###################################################
        # Edit here
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        a=self.forward(Osequence)
        b=self.backward(Osequence)
        s=self.sequence_prob(Osequence)
        prob = a*b/s
        ###################################################
        # Edit here
        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        S = len(self.pi)
        L = len(Osequence)
        path = []
        delta=np.zeros([S,L])
        delta[:,0]=self.pi*self.B[:,self.obs_dict[Osequence[0]]]
        Delta=np.zeros([S,L-1])
        for j in range(1,L):
            for i in range(S):
                delta[i][j]=self.B[i][self.obs_dict[Osequence[j]]]*np.max(self.A[:,i]*delta[:,j-1])
                Delta[i][j-1]=np.argmax(self.A[:,i]*delta[:,j-1])
        ind=np.zeros(L,dtype="int32")
        ind[L-1]=np.argmax(delta[:,L-1])
        for i in reversed(range(L-1)):
            ind[i]=Delta[ind[i+1]][i]
        for i in range(len(ind)):
            path.append(self.key_list[self.val_list.index((ind[i]))])
        ###################################################
        # Edit here
        ###################################################
        return path
