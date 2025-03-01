#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:08:06 2024

Reference implementation for checking WAXp condition.

@author: jpms
"""

#------------------------------------------------------------------------------
#
class CXpMatrix:

  #----------------------------------------------------------------------------
  #
  def __init__(self, _matrix):
    self.matrix = _matrix
    (self.nrow, self.ncol) = _matrix.shape

  #----------------------------------------------------------------------------
  #
  def is_weak_axp(self, vect):
    mat = self.matrix
    (nrow, _) = mat.shape
    for row in range(nrow):    ## To optimize, e.g. using NumPy
      hits = False
      for col in vect:         ## To optimize, e.g. using NumPy
        if mat[row,col-1]:
          hits = True
          break
      if not hits:
        return False
    return True

#------------------------------------------------------------------------------
