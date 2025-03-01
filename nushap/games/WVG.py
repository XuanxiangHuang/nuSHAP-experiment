#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:50:21 2024

An implementation of a weighted voting game (WVG), which is used for
testing purposes in the context of estimating Shapley values.

@author: jpms
"""

import sys
import os

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from games import GameRef


#------------------------------------------------------------------------------
#
class WVG(GameRef.Game):
  '''
    Class representing a weighted voting game (WVG), which represents
    a special case of a simple game.
    '''
  #----------------------------------------------------------------------------
  #
  def __init__(self, qval, wvec):
    super().__init__(len(wvec))
    self.qval = qval
    self.wvec = wvec

  #----------------------------------------------------------------------------
  #
  def cf(self, vec):
    sum, vlen = 0, len(vec)
    for i in range(vlen):
      sum += self.wvec[vec[i]-1]
    return sum >= self.qval

  #----------------------------------------------------------------------------
  #
  def __str__(self):
    return "q: %d  ;  w[]: [ %s ]" % \
            (self.qval, ', '.join(map(str, self.wvec)))          

#------------------------------------------------------------------------------
