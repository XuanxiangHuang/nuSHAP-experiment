#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:28:38 2024

The svestim module implements a Shapley value estimator, using
the algorithm of Castro, Gomez & Tejada (CGT), COR'09. For testing
purposes, the characteristic function of weighted voting games is
considered. Thus, the implementation also illustrates the use of
svestim in the case of estimating the Shapley values in the case of
weighted voting games (WVGs). In more general settings, svestim is
expected to be used as a module, where the user provides the target
characteristic function, by specifying a simple game. 

@author: jpms
"""

import sys
import os

import math
import random
import statistics

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from utils import config
from utils import times
from utils import rand
from utils import nums

from parsers import GameParser


#------------------------------------------------------------------------------
#
# Module-level variables ...

#------------------------------------------------------------------------------
#
class SvEstimator:
  
  def __init__(self, sgame, mval=None):
    # Variables used by the CGT algorithm
    self.alpha = config.alpha
    self.error = config.error
    self.mval = mval
    self.game = sgame
    self.ndist = statistics.NormalDist(0, 1) 

  #----------------------------------------------------------------------------
  #
  def calc_sigsq(self):
    mxdiff = 0
    game = self.game
    nelem = game.NElem
    for i in range(1, nelem+1):
      diff = game.maxval(i) - game.minval(i)
      if diff > mxdiff:
        mxdiff = diff
    rval = mxdiff * mxdiff / 4
    return rval

  #----------------------------------------------------------------------------
  #
  def calc_mval(self):
    za2 = self.ndist.inv_cdf(1-self.alpha)
    ssq = self.calc_sigsq()
    ##print('za2: %.2f ssq: %.2f' % (za2, ssq))
    self.mval = int(round(za2 * za2 * ssq / (self.error * self.error), 0))
    if config.verbosity >= 10 or config.debug:
      print('### mval:', self.mval)
    return self.mval

  #----------------------------------------------------------------------------
  #
  def calc_error(self):  # If mval specified, and error not...
    za2 = self.ndist.inv_cdf(1-self.alpha)
    ssq = self.calc_sigsq()
    self.error = math.sqrt(za2*za2*ssq/self.mval)
    return self.error

  #----------------------------------------------------------------------------
  # Reference implementation for computing contribs
  #
  def add_contribs_ref(self, mv, vect, svs):
    game = self.game
    for k in range(game.NElem):
      ordv = vect[:k]
      ordvi = vect[:k+1]
      svs[vect[k]] += game.cf(ordvi) - game.cf(ordv)
  
  #----------------------------------------------------------------------------
  # Default implementation for computing contribs
  #
  def add_contribs_def(self, mv, vect, svs):
    game = self.game
    for i in range(1,game.NElem+1):
      ordv, ordvi = [], []
      for k in range(game.NElem):
        if vect[k] == i:
          ordv = vect[:k]
          ordvi = vect[:k+1]
          break
      svs[i] += (game.cf(ordvi)-game.cf(ordv))

  #----------------------------------------------------------------------------
  # Reference implementation of Sv estimator
  #
  def run_estimator_ref(self, svs):
    vect, mv = self.game.elements().copy(), self.mval
    while mv >= 0:
      random.shuffle(vect)
      ##print('mv: %d  ;  vect: %s' % (mv, vect))
      if config.contrib_mode == 'ref':
        self.add_contribs_ref(mv, vect, svs)
      else:
        self.add_contribs_def(mv, vect, svs)
      mv -= 1

  #----------------------------------------------------------------------------
  #
  def run_estimator(self, svs):
    '''
      Main algorithm
    '''
    if self.mval == None:
      self.calc_mval()
    elif self.error == None:
      self.calc_error()

    self.run_estimator_ref(svs)

    nd = nums.num_dec_places(config.error) + 1
    for i in range(len(svs)):
      res = svs[i] / self.mval
      svs[i] = f"{res:.{nd}f}"

#------------------------------------------------------------------------------
# Specification of test cases
#
class TCs:
  tc = {}
  tc['01'] = '[7 ; 5, 5, 2, 1]'
  tc['02'] = '[12 ; 4, 4, 4, 2, 2, 1]'
  tc['03'] = '[41; 10, 10, 10, 10, 5, 5, 3, 3, 2]'
  tc['04'] = '[62; 10, 10, 10, 10, 8, 5, 5, 5, 5, 4, 4, 3, 3, 3, 2]'
  tc['32'] = '[12; 4, 4, 4, 2, 1, 1, 1]'
  tc['33'] = '[6; 4, 2, 1, 1, 1, 1]'
  tc['34'] = '[16; 10, 6, 4, 2, 2]'
  tc['35'] = '[21; 12, 9, 4, 4, 1, 1, 1]'
  tc['36'] = '[21; 12, 9, 4, 4, 1, 1, 1 ]'
  tc['37'] = '[9; 9, 2, 2, 2, 2, 1, 1]'
  tc['38'] = '[16; 10, 6, 4, 2, 2, 1]'

#------------------------------------------------------------------------------
# Running test case(s):
#
def run_test_case(gamedef):
  # 1. Set up
  rand.set_rand_seed(0, config.rand)
  game = GameParser.parse_game(gamedef)
  if config.verbosity >= config.HIGHVERB:
    times.report_elapsed_time('Done with set-up')
  
  # 2. Run estimator
  sve = SvEstimator(game)
  svs = [0] * (sve.game.NElem+1)
  sve.run_estimator(svs)
  print('### Svs: ', end='')
  for i in range(1, sve.game.NElem+1):
    print(svs[i], end=' ')
  print()  

  # 3. Wrap-up
  if config.verbosity >= config.HIGHVERB:
    times.report_total_time('Terminating execution')  

#------------------------------------------------------------------------------
#
if __name__ == "__main__":
  '''
    The main serves *only* for testing purposes
  '''  
  for tc in TCs.tc:
    print('### Running test case:', tc)
    run_test_case(TCs.tc[tc])

#------------------------------------------------------------------------------
