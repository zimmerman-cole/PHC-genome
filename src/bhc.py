"""
Bayesian hierarchical clustering (BHC).

Based on "Bayesian Hierarchical Clustering" by Katherine A. Heller and Zoubin Ghahramani.
https://www2.stat.duke.edu/~kheller/bhcnew.pdf
"""

import itertools
import numpy as np
from scipy.special import gammaln
from .utils import is_iterable

class Node(object):
    
    def __init__(
        self, data, alpha, model, left_child=None, right_child=None, index=None,
    ):
        self.data = data
        self.N = data.shape[0]
        self.alpha = alpha
        self.log_alpha = np.log(alpha)
        self.model = model
        self.index = index
        
        self.log_pr_data_h1 = model.log_marginal_likelihood(data)
        
        self.left_child = left_child
        self.right_child = right_child
        if left_child is not None and right_child is not None:
            self.log_d = np.logaddexp(
                self.log_alpha + gammaln(self.N),
                self.left_child.log_d + self.right_child.log_d
            )
            
            self.log_pi = self.log_alpha + gammaln(self.N) - self.log_d
            
            log_1p = np.log(1. - np.exp(self.log_pi)) # = log(1 - pi_k)
            lpt_left = left_child.log_pr_data_tk
            lpt_right = right_child.log_pr_data_tk
            
            post_pr_h1 = self.log_pr_data_h1 + self.log_pi
            
            self.log_pr_data_tk = np.logaddexp(
                post_pr_h1, log_1p + lpt_left + lpt_right
            )
            
            self.log_rk = post_pr_h1 - self.log_pr_data_tk 
            
    @classmethod
    def merge(cls, left_child, right_child):
        assert left_child.alpha == right_child.alpha
        assert left_child.model == right_child.model
        
        data = np.vstack([left_child.data, right_child.data])
        index = frozenset().union(left_child.index, right_child.index)
        return cls(
            data, left_child.alpha, left_child.model, left_child, right_child, index
        )


class Leaf(Node):
    
    def __init__(self, data, alpha, model, index=None):
        if not is_iterable(index):
            index = [index]
        super(Leaf, self).__init__(data, alpha, model, None, None, index)
        
        self.log_d = self.log_alpha
        self.log_pi = 0.             # log(1)
        
        self.log_pr_data_tk = self.log_pr_data_h1
        self.log_rk = 0.             # pi_k P(D_k | H_1^k) / pi_k P(D_k | H_1^k) = 1; log(1) = 0
        
                
class BHC(object):
    
    def __init__(self, data, alpha, model, indices=None):
        self.alpha = alpha
        self.model = model
        self.N = data.shape[0]
        
        if indices is None:
            indices = list(range(data.shape[0]))
        else:
            assert is_iterable(indices) and len(indices) == self.N
        
        self.nodes = dict()
        self.leaves = []
        self.root = None
        
        for idx, data_point in zip(indices, data):
            leaf = Leaf(data_point, alpha, model, frozenset([idx]))
            self.leaves.append(leaf)
            self.nodes[idx] = leaf
            
    def build_tree(self, inner_hooks=list(), outer_hooks=list()):
        to_merge = self.leaves.copy()
        
        i = 0
        while len(to_merge) > 1:
            candidates = dict()
            for j, (node1, node2) in enumerate(itertools.combinations(to_merge, 2)):
                node = Node.merge(node1, node2)
                candidates[node] = (node.log_rk, node1, node2)
                
                for hook in inner_hooks:
                    hook(node, node1, node2, j)
                
            best = sorted(candidates.items(), key=lambda itm: itm[1][0])[-1]
            new_node, (log_rk, left, right) = best
            self.nodes[new_node.index] = new_node    
            to_merge.remove(left)
            to_merge.remove(right)
            to_merge.append(new_node)
            
            for hook in outer_hooks:
                hook(new_node, left, right, i)
                
            i += 1
            
        self.root = new_node
  
        
        
        
        
        
        
        
        
        
        
