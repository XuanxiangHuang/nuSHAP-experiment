#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:20:43 2024

Basic random utils.

@author: jpms
"""

import random

#------------------------------------------------------------------------------
#
def set_rand_seed(seed, truerand=False):
  seed = None if truerand else seed
  random.seed(seed)

#------------------------------------------------------------------------------
