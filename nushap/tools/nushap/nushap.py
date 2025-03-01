#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:19:20 2024

Prototype of a SHAP replacement that corrects the misconceptions
of the original SHAP tool.

@author: jpms
"""

import sys
import os

import argparse
import textwrap

script_dir_name = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(script_dir_name) + '/..'
sys.path.append(root_dir_name)

from utils import config
config.init(script_dir_name)    # Must run init after importing from main

from utils import times, rand, sigs #, report

from tabxp import SbXpRef
from tabxp import CXpMatRef
from svestim import SvEstimRef
from tabxp import XpGameRef
from parsers import TabularDataRef as td

#------------------------------------------------------------------------------
# Local variables

COPYRIGHT = open("LicenseCopyright/COPYRIGHT").read()

#------------------------------------------------------------------------------
#
def parse_data():
  # 1. Parse dataset (restricted set)
  tab = td.parse_tabular_data(config.dataset)
  if config.debug:
    print('Input table:\n', tab)
  # 2. Parse instance (from file)
  inst = td.parse_tabular_data(config.instfile)
  if config.debug:
    print('Target instance:\n', inst)
  if config.verbosity >= config.HIGHVERB:
    times.report_elapsed_time('Done parsing dataset & instance')
  return tab, inst

#------------------------------------------------------------------------------
#
def parse_args():
  """
      Parses command-line options.
  """
  parser = argparse.ArgumentParser(
      prog=os.path.basename(__file__), #'sbxp',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent(COPYRIGHT))
  parser.add_argument('-d', '--debug', help='activate debug mode', dest='debug',
                      action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('-v', '--verbosity', type=int, help='set verbosity level',
                      dest='verbosity', default=0)
  #
  parser.add_argument('-cid', '--colid', dest='colid',
                      help='set to true if ID columns exists', 
                      action=argparse.BooleanOptionalAction,
                      default=config.colid)
  parser.add_argument('-xc', '--exclude', nargs='+', dest='excludes',
                      help='columns to exclude', default=config.excludes)
  parser.add_argument('-trgt', '--target', dest='target',
                      help='target column', default=config.target)
  #
  parser.add_argument('--alpha', type=float, help='value of alpha',
                      dest='alpha', default=config.alpha)
  parser.add_argument('--error', type=float, help='value of error',
                      dest='error', default=config.error)
  parser.add_argument('--cmode', type=str, dest='contrib_mode',
                      help='contrib mode', default=config.contrib_mode)
  #
  parser.add_argument('--rand', help='run in true random', dest='randomize',
                      action=argparse.BooleanOptionalAction,
                      default=config.randomize)
  parser.add_argument('--seed', type=int, help='set random seed', dest='seed',
                      default=config.seed)
  #
  requiredArgs = parser.add_argument_group('required named arguments')
  requiredArgs.add_argument('-ds', '--dataset', type=str, dest='dataset',
                            help='input dataset', required=True)
  requiredArgs.add_argument('-if', '--instfile', dest='instfile',
                            type=str, help='file with instance', required=True)
  opts = parser.parse_args()
  return opts

#------------------------------------------------------------------------------
#
def main():
  # 1a. Parse command line arguments
  opts = parse_args()
  config.reconfig(vars(opts))  # Access namespace as dictionary
  # 1b. Additional configurations
  if config.randomize and not config.true_rand:
    rand.set_rand_seed(config.seed)
  sigs.set_interrupt_handler()
  ##report.redirect_error()
  if config.debug:
    print('Opts:', opts)
    print('Vars:',vars(opts))
  # 1c. Print copyright notice
  print(COPYRIGHT)
  if config.verbosity >= config.HIGHVERB:
    times.report_elapsed_time('Done parsing arguments')

  # 2. Parse dataset(s)
  tab, inst = parse_data()

  # 3. Create an explanation game
  sbxp = SbXpRef.SbXp(tab, inst)
  sbxp.construct_wcxp_matrix()
  sbxp.find_all_cxps()  # Why? Because this will speed up checking WAXps
  mat = sbxp.extract_cxp_matrix_np()

  cxpm = CXpMatRef.CXpMatrix(mat)
  if config.debug and False:
    print('CXpMat:', cxpm.matrix)
  xpg = XpGameRef.XpGame(cxpm, cxpm.ncol)
  if config.verbosity >= config.HIGHVERB:
    times.report_elapsed_time('Done setting up AXp/CXp info')

  # 4. Run Sv estimator
  sve = SvEstimRef.SvEstimator(xpg)
  svs = [0] * (xpg.NElem+1)
  sve.run_estimator(svs)

  # 5. Wrap-up
  print('### Svs: ', end='')
  for i in range(1, sve.game.NElem+1):
    print(svs[i], end=' ')
  print()  
  if config.verbosity >= config.HIGHVERB:
    times.report_total_time('Terminating execution')  

#------------------------------------------------------------------------------
#
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------
