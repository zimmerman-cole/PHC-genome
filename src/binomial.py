"""
Binomial-based models of genetic data.
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.special import comb, gammaln, logsumexp
from scipy.stats import beta as beta_dist


class BinomialModel(ABC):
    """
    Base class for a binomial-based model of P(D_k | H_1^k), i.e. the (marginal) probability 
      that data D_k was generated from a single cluster, with binomial parameters theta 
      marginalized away.
    """
    
    def __init__(self):
        pass
    
    def log_likelihood(self, data, theta, eps=1e-5):
        """
        Compute log P(D_k | theta) for some parameter settings theta.
        
        Returns a LIST of length S, where S = theta.shape[0] 
        (S = the number of parameter settings to calculate the likelihood of).
        """
        if len(theta.shape) == 1:
            theta = theta.reshape(1, -1)

        S = theta.shape[0]
        N, M = data.shape
        
        C = np.log(comb(N=2, k=data)).sum()
        
        # check for any theta that equal 0 or 1, add/subtract a small constant
        # (to avoid log(0) errors)
        idx = np.argwhere(np.isclose(theta, 0))
        theta[idx[:,0], idx[:,1]] += eps
        idx = np.argwhere(np.isclose(theta, 1))
        theta[idx[:,0], idx[:,1]] -= eps
        
        warnings.filterwarnings('error')
        try:
            T1 = np.dot(data, np.log(theta).T).sum(axis=0)
            T2 = np.dot((2 - data), np.log(1 - theta).T).sum(axis=0)
        except RuntimeWarning:
            print(theta)
            raise
        finally:
            warnings.filterwarnings('default')
        
        return C + T1 + T2
    
    @abstractmethod
    def log_marginal_likelihood(self, *args, **kwargs):
        pass


class BetaBinomial(BinomialModel):
    
    """
    When X ~ BetaBinomial(n, alpha, beta):
    P(X = k | n, alpha, beta) = G(n+1) / (G(k+1) G(n - k + 1))
                              * (G(k+alpha) G(n - k + beta)) / G(n + alpha + beta)
                              * G(alpha + beta) / (G(alpha) G(beta))
    where G is the gamma function, and 
    P(X = k | n, alpha, beta) = \int P(X = k | n, p) P(p | alpha, beta) dp
    
    See https://en.wikipedia.org/wiki/Beta-binomial_distribution
    """
    
    def __init__(self, alpha, beta, n=2):
        super(BetaBinomial, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.n = n
        
        # terms in the marginal log likelihood equation that are constant 
        # w.r.t. the data
        self.const = gammaln(n+1) - gammaln(n + alpha + beta) + gammaln(alpha + beta)
        self.const -= gammaln(alpha) + gammaln(beta)
        
    def log_marginal_likelihood(self, X):
        """
        TODO: doc
        """
        alpha, beta, n = self.alpha, self.beta, self.n
        
        # terms that are NOT constant w.r.t. X
        terms = - gammaln(X + 1) - gammaln(n - X + 1) + gammaln(X + alpha)
        terms = terms + gammaln(n - X + beta)
        
        return (terms + self.const).sum()    
    
    
class BetaPriorMC(BinomialModel):
    
    """
    Approximates P(D_k | H_1^k) = \int P(D_k | theta) P(theta | beta) dtheta
    using Monte Carlo integration.
    Sampled theta are drawn from a beta distribution.
    """
    
    def __init__(self, alpha, beta, num_samples=10000):
        self.alpha = alpha
        self.beta = beta
        self.num_samples = num_samples
        
    def log_marginal_likelihood(self, X, include_MLE=False):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        sampled_params = np.random.beta(
            self.alpha, self.beta, size=(self.num_samples, X.shape[1])
        )
        if include_MLE:
            theta_MLE = (X.mean(axis=0) / 2.).reshape(1, -1)
            sampled_params = np.vstack([sampled_params, theta_MLE])
        
        log_priors = np.log(
            beta_dist.pdf(sampled_params, self.alpha, self.beta)
        ).sum(axis=1)
        log_priors -= log_priors.max()

        log_liks = self.log_likelihood(X, sampled_params)
        
        # log of unnormalized posterior probabilities
        # p(X | theta) p(theta | alpha, beta)
        log_posts = log_priors + log_liks
        
        lml = logsumexp(log_posts)
        return lml


class BinomialMLE(BinomialModel):
    
    """
    Approximates P(D_k | H_1^k) = \int P(D_k | theta) P(theta | beta) dtheta
    by just using the maximum likelihood estimate (MLE) of theta.
    
    WARNING: using this model often results in undesirable behaviour during the early stages 
      of clustering, especially for clusters with only one data point.
      TODO: explain
    """
    
    def __init__(self):
        pass
    
    def log_marginal_likelihood(self, X, eps=1e-5):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        theta_MLE = X.mean(axis=0) / 2.
        
        # smooth theta_MLE, so it has no parameters exactly equal to 0 or 1
        ix0 = np.argwhere(np.isclose(theta_MLE, 0.))
        theta_MLE[ix0] += eps
        ix1 = np.argwhere(np.isclose(theta_MLE, 1.))
        theta_MLE[ix1] -= eps
        
        C = np.log(comb(N=2, k=X)).sum()
        
        T1 = X * np.log(theta_MLE)
        T2 = (2 - X) * np.log(1 - theta_MLE)
        
        out = (C + T1 + T2).sum()
        return out
    
    
class BinomialExperimental(BinomialModel):
    
    def __init__(self):
        pass
    
    def log_marginal_likelihood(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        N, M = X.shape
        
        theta_MLE = (X.mean(axis=0) / 2.).reshape(1, M)
    
        theta_hlf = np.array([0.5]*M).reshape(1, M)
    
        theta = np.vstack([theta_MLE, theta_hlf])
    
        return self.log_likelihood(X, theta).mean()
    
    
    
    
    
    
    
    
    
    
    
    
    