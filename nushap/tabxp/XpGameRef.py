#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:01:41 2024

Implements the characteristic function of the explanation game.

@author: jpms
"""

import sys
import os

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from games import GameRef

#------------------------------------------------------------------------------
#
class XpGame(GameRef.Game):

  #----------------------------------------------------------------------------
  #
  def __init__(self, cxpm, nf):
    self.cxpm = cxpm
    super().__init__(nf)    # Matrix is assumed non-empty

  #----------------------------------------------------------------------------
  #
  def cf(self, vec):
    '''
      For the explanation game, cf tests whether picked features are a WAXp.
      Checking whether the picked features are a WCXp is another option.
    '''
    return self.cxpm.is_weak_axp(vec)

  #----------------------------------------------------------------------------
  #
  def __str__(self):
    return 'CXps:\n%s' % (self.cxpm)

#------------------------------------------------------------------------------
