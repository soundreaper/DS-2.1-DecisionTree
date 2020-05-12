import numpy as np
import pandas as pd


class DecisionTreeNode(object):
    def __init__(self, name, pointers=None):
        self.name = name
        if pointers is not None and type(pointers) == list():
            self.pointers = pointers
        else:
            self.pointers = list()

    def __repr__(self):
        return f'Node({self.name})'

    def is_leaf(self):
        """Return True if this node is a leaf (has no children)."""
        return self.pointers == []

    def is_branch(self):
        """Return True if this node is a branch (has at least one child)."""
        return self.pointers == [] and type(self.pointers) == list()

    def add_pointer(self, pointer):
        """Use this to add to a given nodes pointers list. Format as
        ('name', node)."""
        self.pointers.append(pointer)

    def height(self):
        """Return the height of this node (the number of edges on the longest
        downward path from this node to a descendant leaf node)."""
        return max([pointer[1].height() + 1 for pointer in self.pointers]) if len(self.pointers) >= 1 else 0


class DecisionTree(object):

    def __init__(self, max_depth):
        """Initialize this tree"""
        self.root = None
        self.size = 0
        self.max_depth = max_depth

    def __repr__(self):
        """Return a string representation of this tree."""
        return f'desicionTree({self.size} nodes)'

    def is_empty(self):
        """Return True if this tree is empty (has no nodes)."""
        return self.root is None

    def height(self):
        """Return the height of this tree (the number of edges on the longest
        downward path from this tree's root node to a descendant leaf node)."""
        return self.root.height()

    def items_level_order(self):
        """Return a level-order list of all items in this tree."""
        items = []
        if not self.is_empty():
            self._traverse_level_order_iterative(self.root, items.append)
        return items

    def _traverse_level_order_iterative(self, start_node, visit):
        """iterative level-order traversal"""
        queue = []
        queue.append(start_node)
        while queue is not []:
            node = queue.pop(0)
            visit([node.name, [pointer[0] for pointer in node.pointers]])

            for pointer in node.pointers:
                queue.append(pointer[1])

    def fit(self, df, target):
        """Function stub, Calculate entropy of each column and add nodes to the
        tree."""
        pass

    def entropy(self, column, target, given=None):
        """Func stub, clac entropy helper function."""
        pass

    def predict(self, df):
        """Given a df, predict each row's label"""
        pass
