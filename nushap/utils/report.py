#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:07:13 2024

Different levels of reporting, with actions taken in applicable.

@author: jpms
"""

import sys

from utils import config

#------------------------------------------------------------------------------
#
def redirect_output(fname=None):
  if fname == None:
    fname = config.logfile
  fref = open(fname, 'w')
  sys.stdout = sys.stderr = fref

#------------------------------------------------------------------------------
#
def redirect_error(fname=None):
  if fname == None:
    fname = config.logfile
  fref = open(fname, 'w')
  sys.stderr = fref

#------------------------------------------------------------------------------
#
def prt_log_msg(msg, level='ERROR', critical=True):
  print('## ***%s***: %s' % (level, msg))
  if critical:
    sys.exit(-1)

#------------------------------------------------------------------------------
#
def prt_info(msg, critical=True):
  prt_log_msg(msg, 'INFO', critical)

#------------------------------------------------------------------------------
#
def prt_warn(msg, critical=True):
  prt_log_msg(msg, 'WARN', critical)
  
#------------------------------------------------------------------------------
#
def prt_error(msg, critical=True):
  prt_log_msg(msg, 'ERROR', critical)

#------------------------------------------------------------------------------
