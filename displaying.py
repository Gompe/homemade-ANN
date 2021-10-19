# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:20:38 2021

@author: gomes
"""

import graphviz

def display(self):
    dot = graphviz.Digraph(comment='Multilayer Perceptron')
    for i in range(self.layers):
        for j in range(self.sizes[i]):
            if i == 0:
                dot.node(str((0,j)), '0')
            else:
                dot.node(str((i,j)), str(self.biasses[i-1][0][j]))
                if i < self.layers - 1:
                    for k in range(self.sizes[i-1]):
                        dot.edge(str((i-1,k)), str((i,j)),
                                 str(self.weightMatrices[i-1][j][k]))
    print(dot.source)
    dot.render('MLP.gv', view=True)