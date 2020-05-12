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
        while queue != []:
            node = queue.pop(0)
            visit([node, [pointer for pointer in node.pointers]])

            for pointer in node.pointers:
                queue.append(pointer[1])

    def fit(self, df, target):
        """Function stub, Calculate entropy of each column and add nodes to the
        tree."""
        self.root = DecisionTreeNode(self.max_info_gain(df, target)[1])
        self.size += 1
        self._recursive_fit(df, target, self.root)



    def _recursive_fit(self, df, target, parent_node):
        s = df[parent_node.name].unique()
        for i in s:
            new_df = df[df[parent_node.name] == i].drop(columns=parent_node.name)
            gain_name = self.max_info_gain(new_df, target)
            if gain_name[0] == 0:
                final = self.conditional_prob(df, parent_node.name, target, i)
                new_node = DecisionTreeNode(list(final.keys())[0])
                self.size += 1
                parent_node.pointers.append((i, new_node))
            else:
                new_node = DecisionTreeNode(gain_name[1])
                parent_node.pointers.append((i, new_node))
                self.size += 1
                self._recursive_fit(new_df, target, new_node)


    def _entropy(self, p):
        """Func stub, clac entropy helper function."""
        H = np.array([-i*np.log2(i) for i in p]).sum()
        return H

    def conditional_prob(self, df, c1, c2, condition):
        df_new = df[df[c1] == condition][c2]
        s = df_new.unique()
        population_size = len(df_new)
        pr = {}
        for i in s:
            pr[i] = len(df[(df[c1] == condition) & (df[c2]== i)]) / population_size
        return pr

    def probability(self, df, col):
        s = df[col].unique()
        pr = {}
        for i in s:
            pr[i] = len(df[df[col] == i]) / len(df[col])
        return pr

    def info_gain(self, df, feature, target):
        # obtain the entropy of the decision
        dict_decision = dict(df[target].value_counts())
        prob_decision = [q for (p,q) in dict_decision.items()]/sum(dict_decision.values())
        entropy_decision = self._entropy(prob_decision)
    #     print(entropy_decision)

        # obtain the probabilities of the feature
        # example: for Wind, obtain the probabilities of Strong and Weak
        dict_feature = dict(df[feature].value_counts())
        dict_prob_feature = {}
        for (p,q) in dict_feature.items():
            dict_prob_feature[p] = q/sum(dict_feature.values())
    #     print(dict_prob_feature)

        # obtain the probability of the decision,
        # for all possible values of the feature (conditions)
        conditions = df[feature].unique()
        dict_ = {}
        for condition in conditions:
            dict_[condition] = self.conditional_prob(df, feature, target, condition)
    #     print(dict_)

        # Given the above metrics, calculate the information gain
        # between the feature and the decision using the formula we learned
        S = 0
        for (i,j) in dict_.items():
    #         print(i,j)
            prob_condition = list(dict_[i].values())
    #         print(entropy_condition)
            S = S + dict_prob_feature[i]*self._entropy(prob_condition)
    #         print(dict_prob_feature[i]*entropy(entropy_condition))
        return entropy_decision - S


    def max_info_gain(self, df, target, givens=None):
        if givens is not None:
            for col, value in givens:
                df = df[df[col] == value]
                df = df.drop(columns=[col])
        info_gains = [(self.info_gain(df, column, target), column) for column in df.columns[0:-1]]
        highest = info_gains[0]
        # print(info_gains)
        for this_info_gain in info_gains:
            if this_info_gain[0] > highest[0]:
                highest = this_info_gain
        return highest

    def predict(self, df):
        """Given a df, predict each row's label"""
        pass
