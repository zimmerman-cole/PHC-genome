"""
Visualize constructed trees. 
"""

import numpy as np
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node


def show_tree(bhc):
    """
    Visualize Bayesian hierarchical clustering (BHC) tree using ete3.
    """

    t = Tree()

    node2parent = {bhc.root: t}

    nodes = [bhc.root]
    
    tags = set(bhc.root.index)

    while len(nodes) > 0:
        node = nodes.pop(0)
        uniques, counts = np.unique(node.index, return_counts=True)
        purity = counts.max() / counts.sum()

        if len(node.index) > 1:
            name = 'n_k=%d; purity=%.2f; p=%.2f, lp=%.2f' \
                        % (len(node.index), purity, np.exp(node.log_rk), node.log_rk)
        else:
            name = '%s (%s)' % (node.index[0], node.tags['Geographic origin'])

        C = node2parent[node].add_child(name=name)

        if node.left_child is not None:
            assert node.right_child is not None

            nodes.extend([node.left_child, node.right_child])
            node2parent[node.left_child] = C
            node2parent[node.right_child] = C

    ts = TreeStyle()
    ts.show_leaf_name = False
    def my_layout(node):
        F = TextFace(node.name, tight_text=True)
        add_face_to_node(F, node, column=0, position="branch-right")
    ts.layout_fn = my_layout
    # t.render('out.svg', tree_style=ts)
    # ts.mode = 'c'

    t.show(tree_style=ts)
    
    return t
    
t = show_tree(bhc)
# t.render('tree_BinomialExperimental.pdf')
