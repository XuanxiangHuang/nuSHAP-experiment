#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:17:36 2024

Load YAML configuration file and initialize module variables.
The module allows for module variables to be redefined given a
set of user-defined arguments. See use of function 'reconfig'.
Note: all configuration are specified either in the defaults.yml
file, or otherwise when recondiguring.
Implementation is common to other tools.

@author: jpms
"""
import sys
import os

import yaml

script_dir_name = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(script_dir_name)
sys.path.append(root_dir_name)

#------------------------------------------------------------------------------
# NOTE: init() MUST be executed when module is loaded by running script!!!
#------------------------------------------------------------------------------
# Constants
#
NOVERB = 0
LOWVERB = 2
MEDVERB = 5
HIGHVERB = 10

REFID = '_RefID_'

tooldir = None
cfg = None

#------------------------------------------------------------------------------
#
def load_config():
  assert tooldir != None
  with open(tooldir + '/defaults.yml', 'r') as inp:
    cfg = yaml.safe_load(inp)
  inp.close()
  ##print(cfg)
  return cfg

#------------------------------------------------------------------------------
#
def initialize(cfg):
  globals().update(cfg)

#------------------------------------------------------------------------------
#
def reconfig(optvars):
    for arg in optvars:
      ##print('arg:', arg)
      globals()[arg] = optvars[arg]
    # This is more specific; could be defined elsewhere. For now, the
    # option is to all globals in the config file and any additional 
    # ones in here.
    if 'actions' in globals():
      globals()['actionset'] = set(globals()['actions'])

#------------------------------------------------------------------------------
# NOTE: init() MUST be executed when module is loaded by running script!!!
#
def init(tdir):
  global cfg
  global tooldir
  tooldir = tdir
  cfg = load_config()
  globals().update(cfg)

##cfg = load_config()
##initialize(cfg)

#------------------------------------------------------------------------------
#
def main():
  init()
  print(cfg)
  print(verbosity)
  print(debug)
  print(actions)

#------------------------------------------------------------------------------
#
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------
