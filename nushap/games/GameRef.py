#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:46:01 2024

Abstract definition of a simple game.

@author: jpms
"""

#------------------------------------------------------------------------------
#
class Game:
  '''
    Abstract Game class
  '''
  #----------------------------------------------------------------------------
  #
  def __init__(self, nelem):
    self.NElem = nelem                   # Number of elements
    self.Elems = list(range(1,nelem+1))  # Elements numbered 1 to NElem

  #----------------------------------------------------------------------------
  #
  def cf(self, vec):
    '''
      For Game, cf (i.e. characterisfic function) is a virtual method.
    '''
    raise NotImplementedError()

  #----------------------------------------------------------------------------
  #
  def elements(self):
    return self.Elems

  #----------------------------------------------------------------------------
  #
  def maxval(self, elem):
    return 1
  
  #----------------------------------------------------------------------------
  #
  def minval(self, elem):
    return 0

#------------------------------------------------------------------------------
