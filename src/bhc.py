"""
Bayesian hierarchical clustering (BHC).

Based on "Bayesian Hierarchical Clustering" by Katherine A. Heller and Zoubin Ghahramani.
https://www2.stat.duke.edu/~kheller/bhcnew.pdf
"""

import itertools
import numpy as np
from scipy.special import gammaln, comb
from .utils import is_iterable
import warnings

class Node(object):
    
    """
    A node in the (binary) Bayesian hierarchical clustering (BHC) tree.
    See https://www2.stat.duke.edu/~kheller/bhcnew.pdf for details.
    
    Arguments:
    ======================================================================================
      (np.array) data: 
      Data D_k for this node (NOT the full data).
    ======================================================================================
      (int) row_idx:
      The row(s) of this node's data in the full data matrix.
    ======================================================================================
      (float) alpha:
      The setting of the (Chinese restaurant process) hyperparameter alpha.
    ======================================================================================
      (Model) model:
      A probabilistic model which can be used to calculate P(D_k | H_1^k) - the (marginal) 
      probability that this node's data was generated from the SAME probabilistic model;
         P(D_k | H_1^k) = \int P(D_k | theta) P(theta | beta) dtheta
      where theta are the model parameters, and beta the hyperparameters.
    ======================================================================================
      (Node) left_child:
      If this node is not a leaf, then this is the left child of this node.
    ======================================================================================
      (Node) right_child:
      If this node is not a leaf, then this is the right child of this node.
    ======================================================================================
      (anything) index:
      (optional) The index/unique ID for this node. 
    ======================================================================================
      (dict) tags:
      (optional) A dictionary containing any extra information associated with this node.
    ======================================================================================
    """
    
    def __init__(
        self, data, row_idx, alpha, model, left_child=None, right_child=None, index=None
    ):
        self.row_idx = row_idx
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
            
#             print(self.index)
#             print('log P(D_k | H_1^k):', self.log_pr_data_h1)
#             print('children:          ', 
#                   self.left_child.log_pr_data_h1 + self.right_child.log_pr_data_h1
#                  )
#             print('pi_k:              ', np.exp(self.log_pi))
#             print('log P(D_k | T_k  ):', self.log_pr_data_tk)
#             print('r_k:               ', np.exp(self.log_rk))
#             input()

    @property
    def tags(self):
        left_tags = self.left_child.tags
        right_tags = self.right_child.tags
        return {
            k: (left_tags[k] + right_tags[k]) for k in left_tags.keys()
        }


class Leaf(Node):
    
    def __init__(self, data, row_idx, alpha, model, index=None, tags=dict()):
        if not is_iterable(index):
            index = [index]
        super(Leaf, self).__init__(data, row_idx, alpha, model, None, None, index)
        self._tags = tags
        
        self.log_d = self.log_alpha
        self.log_pi = 0.             # log(1)
        
        self.log_pr_data_tk = self.log_pr_data_h1
        self.log_rk = 0.             # pi_k P(D_k | H_1^k) / pi_k P(D_k | H_1^k) = 1; log(1) = 0
        
    def __str__(self):
        out = "Leaf(index=%s, log_p=%.1f)" % (self.index[0], self.log_pr_data_tk)
        return out
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def tags(self):
        return self._tags

        
class CandidateSet(object):
    
    def __init__(self, candidate_dict):
        self.candidates = candidate_dict
        
    def get_best_candidate(self):
        """
        Returns new_node, (left_child, right_child) with the highest log_rk.
        """
        return max(self.candidates.items(), key=lambda itm: itm[0].log_rk)
    
    def prune(self, node):
        """
        Removes any entries {parent: (left, right)} where node == left or node == right. 
        """
        to_delete = []
        for parent, (left, right) in self.candidates.items():
            if (node == left) or (node == right):
                to_delete.append(parent)
                
        for n in to_delete:
            del self.candidates[n]
            
    def keys(self):
        return list(self.candidates.keys())
    
    def add(self, key, value):
        self.candidates[key] = value
        
                
class BHC(object):
    """
    TODO: doc
    """
    
    def __init__(self, data, alpha, model, indices=None, tags=None):
        self.data = data
        self.alpha = alpha
        self.model = model
        self.N = data.shape[0]
        
        if tags is None:
            tags = [dict()] * self.N
        else:
            assert is_iterable(tags)
            assert len(tags) == self.N
            assert all([isinstance(d, dict) for d in tags])
            assert all([d.keys() == tags[0].keys() for d in tags])
            for t_dict in tags:
                for k, v in t_dict.items():
                    if not is_iterable(v) or isinstance(v, str):
                        t_dict[k] = [v]
        
        if indices is None:
            indices = list(range(data.shape[0]))
        else:
            assert is_iterable(indices), "Data indices need to be array-like."
            msg = "len(indices) (%d) != len(data) (%d)." % (len(indices), self.N)
            assert len(indices) == self.N, msg
        
        self.nodes = []
        self.leaves = []
        self.root = None
        
        for i, (idx, data_point, tag) in enumerate(zip(indices, data, tags)):
            leaf = Leaf(data_point, [i], alpha, model, [idx], tag)
            self.leaves.append(leaf)
            self.nodes.append(leaf)
            
    def build_tree(self, inner_hooks=list(), outer_hooks=list()):
        to_merge = self.leaves.copy()
        np.random.shuffle(to_merge)
        
        i = 0
        while len(to_merge) > 1:
            candidates = dict()
            for j, (node1, node2) in enumerate(itertools.combinations(to_merge, 2)):
                node = self._merge(node1, node2)
                candidates[node] = (node.log_rk, node1, node2)
                
                for hook in inner_hooks:
                    hook(node, node1, node2, j)
                
            best = sorted(candidates.items(), key=lambda itm: itm[1][0])[-1]
            new_node, (log_rk, left, right) = best
            self.nodes.append(new_node)    
            to_merge.remove(left)
            to_merge.remove(right)
            to_merge.append(new_node)
            
            for hook in outer_hooks:
                hook(new_node, left, right, i)
                
            i += 1
            
        self.root = new_node
     
    def build_tree2(self, inner_hooks=list(), outer_hooks=list()):
        to_merge = self.leaves.copy()
        np.random.shuffle(to_merge)
        
        i = 0
        # Set up initial set of candidates (each with n_k=2 data points)
        candidates = dict()
        N_c = comb(len(to_merge), 2)
        for j, (node1, node2) in enumerate(itertools.combinations(to_merge, 2)):
            node = self._merge(node1, node2)
            candidates[node] = (node1, node2)
            for hook in inner_hooks:
                hook(node, node1, node2, j, N_c)
                
#         cs = sorted(list(candidates.keys()), key=lambda n: n.log_rk, reverse=True)
#         for c in cs:
#             print(c.index)
#             print('log P(D_k | H_1^k):', c.log_pr_data_h1)
#             print('children:          ', 
#                   c.left_child.log_pr_data_h1 + c.right_child.log_pr_data_h1
#                  )
#             print('pi_k:              ', np.exp(c.log_pi))
#             print('log P(D_k | T_k  ):', c.log_pr_data_tk)
#             print('r_k:               ', np.exp(c.log_rk))
#             input()
#         return candidates
            
                
        candidate_set = CandidateSet(candidates)
        
        while len(to_merge) > 1:
            best = candidate_set.get_best_candidate()
            new_node, (left, right) = best
            self.nodes.append(new_node)    
            candidate_set.prune(left)   # prune any candidates that have left or right as a child
            candidate_set.prune(right)
            
            for hook in outer_hooks:
                hook(new_node, left, right, i)
            
            to_merge.remove(left)
            to_merge.remove(right)
            for j, node in enumerate(to_merge):
                new_candidate = self._merge(new_node, node)
                candidate_set.add(new_candidate, (new_node, node))
                
                for hook in inner_hooks:
                    hook(new_candidate, new_node, node, j, len(to_merge))
                    
            to_merge.append(new_node)
            
            
                
            i += 1
            
        self.root = new_node
    
    def _merge(self, left, right):
        data = np.vstack([self.data[left.row_idx], self.data[right.row_idx]])
        row_idx = left.row_idx + right.row_idx
        index = left.index + right.index
            
        return Node(
            data, row_idx, self.alpha, self.model, left, right, index
        )
        
  
        
        
        
        
        
        
        
        
        
        
