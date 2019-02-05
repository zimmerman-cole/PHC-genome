"""
Restricted Boltzmann Machine.

See http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf.
"""

import numpy as np
from scipy.special import comb, logsumexp
import warnings

class RBM(object):
    
    def __init__(self, num_hidden, num_visible, std=0.01):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        
        self.H = num_hidden + 1   # +1 is for bias
        self.V = num_visible + 1  # +1 is for bias
        
        self.W = np.random.randn(self.V, self.H) * std
        self.W[:, 0] = 0.  # bias hidden unit
        self.W[0, :] = 0.  # first element of any data vector = 1 (bias element)
        
    def CD(self, v, n, do_update=True, lr=None):
        v = v.reshape(-1, self.num_visible)
        N = v.shape[0]  # number of data vectors / batch size
        
        # Insert bias element into data vector(s)
        v = np.insert(v, 0, values=1, axis=1)
        
        # First, calculate <h_j v_i>_{data}  (positive phase)
        hidden_probs = RBM.sigmoid(np.dot(v, self.W))
        hidden_probs[:, 0] = 1.   # Fix bias unit = 1
        hidden_states = np.where(
            hidden_probs > np.random.rand(N, self.H), 1., 0.
        )
        data_sample = np.dot(v.T, hidden_states)
        
        # Second, calculate <h_j v_i>_{recon} (negative phase)
        for i in range(n):
            visible_probs = RBM.sigmoid(np.dot(hidden_states, self.W.T))
            visible_probs[:, 0] = 1.  # Fix bias element = 1
        
            hidden_probs = RBM.sigmoid(np.dot(visible_probs, self.W))
            hidden_states = np.where(
                np.random.rand(N, self.H) > hidden_probs, 1., 0.
            )
            
        # When computing <h_j v_i>_{recon}, use the PROBABILITIES
        # of hidden state activations instead of the actual states.
        recon_sample = np.dot(visible_probs.T, hidden_probs)
        
        grad = (data_sample - recon_sample) / N
        
        rmse = np.sqrt((v - visible_probs)**2).mean()
        
        if do_update:
            if lr is None:
                lr = 10**-3 * np.mean(np.abs(self.W)) / np.mean(np.abs(grad))
            self.W += lr * grad
            
        return grad, rmse
     
    def transform(self, v):
        """ Compute p(h | v) for some vector(s) v. """
        v = v.reshape(-1, self.num_visible)
        N = v.shape[0]  # number of data vectors / batch size
        
        # Insert bias element into data vector(s)
        v = np.insert(v, 0, values=1, axis=1)
        
        hidden_probs = RBM.sigmoid(np.dot(v, self.W)).reshape(N, self.H)
        return hidden_probs
        
    @classmethod
    def sigmoid(cls, x):
        return 1. / (1. + np.exp(-x))

    def __str__(self):
        return "RBM(num_hidden={nh}, num_visible={nv})".format(
            nh=self.num_hidden, nv=self.num_visible
        )
    
    def __repr__(self):
        out = self.__str__()[:-1]
        out = out + ', id={ID})'.format(ID=id(self))
        return out


class DRBM(object):
    """
    See this paper:
    "Classification using Discriminative Restricted Boltzmann Machines"; Larochelle and Bengio
    http://www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf
    """
    
    def __init__(self, num_hidden, num_visible, num_targets, std=0.01):
        self.H = num_hidden
        self.V = num_visible
        self.T = num_targets

        self.b = np.zeros((self.V, 1))  # visible biases
        self.c = np.zeros((self.H, 1))  # hidden biases
        self.d = np.zeros((self.T, 1))  # target biases
    
        self.W = np.random.randn(self.H, self.V) * std    # (H, V): hidden-visible weights
        self.U = np.random.randn(self.H, self.T) * std    # (H, T):  hidden-target weights
        
    def CD(self, v, y, n, do_update=True, lr=None):
        v = v.reshape(-1, self.V)   # v: (N, V)
        N = v.shape[0]
        y = y.reshape(-1, self.T)   # y: (N, T)
        assert v.shape[0] == y.shape[0], "v.shape[0] != y.shape[0]"
        
        # Data sample (positive phase)
        y_idx = np.argwhere(y == 1)[:, 1]   # (T,  )
        C = np.tile(self.c, reps=[1, N])    # (H, N)
        hidden_probs = self._sigmoid(C + self.U[:, y_idx] + np.dot(self.W, v.T))  # (H, N)
        
        W_grad = np.dot(hidden_probs, v)  # (H, V): first term in (batch) gradient for W
        b_grad = v.sum(axis=0).reshape(self.V, 1)
        c_grad = hidden_probs.sum(axis=1).reshape(self.H, 1)
        d_grad = y.sum(axis=0).reshape(self.T, 1)
        U_grad = np.dot(hidden_probs, y)  # (H, T)
        
        # Reconstructed sample (negative phase)
        B = np.tile(self.b, reps=[1, N])  # (V, N)
        D = np.tile(self.d, reps=[1, N])  # (T, N)
        for i in range(n):
            hidden_states = np.where(
                np.random.randn(self.H, N) > hidden_probs, 1., 0.
            )
            
            visible_probs = self._sigmoid(B + np.dot(self.W.T, hidden_states))  # (V, N)
            
            target_scores = D + np.dot(self.U.T, hidden_states)  # (T, N)
            warnings.filterwarnings('error')
            try:
                target_probs = np.exp(target_scores)
                target_probs /= target_probs.sum(axis=0).reshape(1, N)
            except RuntimeWarning:
                normalizers = logsumexp(target_scores, axis=0).reshape(1, N)
                target_scores -= normalizers
                target_probs = np.exp(target_scores)
            finally:
                warnings.filterwarnings('default')
            
            hidden_probs = self._sigmoid(
                C + np.dot(self.U, target_scores) + np.dot(self.W, visible_probs)
            )  # (H, N)
            
        vr = visible_probs  # (V, N)
        yr = target_probs   # (T, N)
        
        W_grad -= np.dot(hidden_probs, vr.T)  # (H, V): first term in (batch) gradient for W
        b_grad -= vr.sum(axis=1).reshape(self.V, 1)
        c_grad -= hidden_probs.sum(axis=1).reshape(self.H, 1)
        d_grad -= yr.sum(axis=1).reshape(self.T, 1)
        U_grad -= np.dot(hidden_probs, yr.T)  # (H, T)
        
        W_grad /= N
        b_grad /= N
        c_grad /= N
        d_grad /= N
        U_grad /= N
        
        if do_update:
            if lr is None:
                lr = 10**-3 * np.mean(np.abs(self.W)) / np.mean(np.abs(grad))
            
            self.W += lr * W_grad
            self.b += lr * b_grad
            self.c += lr * c_grad
            self.d += lr * d_grad
            self.U += lr * U_grad
            
        return (W_grad, U_grad, b_grad, c_grad, d_grad)
    
    def predict_y(self, v):
        """
        Compute p(y | v) for some batch of vectors v.
        
        Args:
        =======================================================================================
        (np.array) v: An array of shape (N, num_visible_units), where N is the number of 
                      vectors to compute p(y | v) for.
                      
        Returns:
        =======================================================================================
        (np.array) out: An array of shape (N, num_classes), where each element out[n, k]
                        equals p(y_k | x_n).
        """
        v = v.reshape(-1, self.V)
        N = v.shape[0]
        
        out = np.zeros((N, self.T))
        for i in range(N):
            
            t1 = np.tile(self.c, reps=[1, self.T])
            t3 = np.dot(self.W, v[i].reshape(self.V, 1))
            t3 = np.tile(t3, reps=[1, self.T])
            # exponent term
            exp = t1 + self.U + t3  # (H, T)
            
            warnings.filterwarnings('error')
            try:
                log_probs = self.d + np.log1p(np.exp(exp)).sum(axis=0).reshape(self.T, 1)
                log_probs -= logsumexp(log_probs)  # normalize
            except RuntimeWarning:
                raise
            finally:
                warnings.filterwarnings('default')
            
            out[i] = np.exp(log_probs).squeeze()
            
        return out  
    
    def predict_y2(self, v):
        v = v.reshape(-1, self.V)     # (N, V)
        N = v.shape[0]
        y = np.diag(np.ones(self.T))  # (T, T)
        
        out = np.zeros((N, self.T))
        for i in range(N):
            V = np.tile(v[i].reshape(1, self.V), reps=[self.T, 1])  # (T, V)
            
            energies = np.zeros(self.T)
            for t in range(self.T):
                energies[t] = self.free_energy(V[t], y[t])
            
            energies *= -1.
            
            energies -= logsumexp(energies)
            out[i] = np.exp(energies)       # probabilities
            
        return out
            
    def free_energy(self, v, y):
        msg = "Method currently only works for single data points."
        assert v.reshape(-1, self.V).shape[0] == 1, msg
        v = v.reshape(self.V, )
        y = y.reshape(self.T, )
        
        bx = -np.dot(self.b.squeeze(), v)
        dy = -np.dot(self.d.squeeze(), y)
        
        t3 = self.c.squeeze() + self.W.dot(v) + self.U.dot(y)
        t3 = np.logaddexp(1, t3).sum()
        
        return bx + dy - t3
        
    def compute_hidden_probabilities(self, v, y):
        """ Returns probabilities P(h | v, y). """
        v = v.reshape(-1, self.V)  # (N, V)
        N = v.shape[0]
        y = y.reshape(-1, self.T)  # (N, T)
        assert v.shape[0] == y.shape[0], "v.shape[0] != y.shape[0]"
        
        C = np.tile(self.c, reps=[1, N])  # (H, N)
        hidden_probs = self._sigmoid(C + np.dot(self.U, y.T) + np.dot(self.W, v.T))
        return hidden_probs  # (H, N)
    
    def compute_class_probabilities(self, h):
        """ Compute p(y | h) for some hidden vector h. """
        h = h.reshape(-1, self.H)  # (N, H)
        N = h.shape[0]
        
        D = np.tile(self.d, reps=[1, N])  # (T, N)
        exp = D + np.dot(self.U.T, h.T)   # (T, N)
        denom = logsumexp(exp, axis=0)    # (N,  )
        exp -= denom.reshape(1, N)        # normalize
        
        return np.exp(exp)

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
        
    def __str__(self):
        return "DRBM2(num_hidden={nh}, num_visible={nv}, num_targets={nt})".format(
            nh=self.num_hidden, nv=self.num_visible, nt=self.num_targets
        )
        
    def __repr__(self):
        out = self.__str__()[:-1]
        out = out + ', id={ID})'.format(ID=id(self))
        return out   
    


    
    
    
    
    
    
    
    
    
    
    
pass
    