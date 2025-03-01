#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:16:55 2024

Implements a weighted voting game (WVG), but introduces some delay,
aiming at simulating the time taken in computing the characteristic
function

@author: jpms
"""

import sys
import os

import time
import random

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from games import WVG

#------------------------------------------------------------------------------
#
class DWVG(WVG.WVG):

  #----------------------------------------------------------------------------
  #
  def __init__(self, qval, wvec,
              defdelay=0, randomize=False):
    super().__init__(qval, wvec)
    self.defdelay = defdelay
    self.randomize = randomize

  #----------------------------------------------------------------------------
  #
  def cf(self, vec):
    # Obs: the command line 'time' command does not count the time in sleep...
    delay = self.defdelay + (len(vec)*random.random() if self.randomize else 0)
    time.sleep(delay)
    return super().cf(vec)

#------------------------------------------------------------------------------
