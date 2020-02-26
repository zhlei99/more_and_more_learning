#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:08:39 2018

@author: zhaolei
"""

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc= "0.8")
leafNode = dict( boxstyle = "round4", fc = "0.8")
arrow_args = dict( arrowstyle = "<-")

def plotNode( nodeTxt, centerPt, parentPt, nodeType ):
    