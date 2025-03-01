#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:34:04 2024

Time-measurement utilities used by other modules.
Implementation is common to all tools.

@author: jpms
"""

import time

#------------------------------------------------------------------------------
# Measure time when module is loaded
#------------------------------------------------------------------------------

start_time = 0
last_time = 0

#------------------------------------------------------------------------------
#
def init_timer():
  global start_time
  global last_time
  start_time = time.time()
  last_time = start_time

#------------------------------------------------------------------------------
#
init_timer()

#------------------------------------------------------------------------------
#
def total_time():
  return time.time() - start_time

#------------------------------------------------------------------------------
#
def report_total_time(msg, prec=3):
  tott = total_time()
  tstr = f"{tott:.{prec}f}" 
  print("### %s. Total running time: %s sec" % (msg, tstr), flush=True)

#------------------------------------------------------------------------------
#
def elapsed_time():
  global last_time
  new_time = time.time()
  elapsed_time = new_time - last_time
  last_time = new_time
  return elapsed_time

#------------------------------------------------------------------------------
#
def report_elapsed_time(msg, prec=3):
  diff = elapsed_time()
  tstr = f"{diff:.{prec}f}" 
  print("### %s. Elapsed time: %s sec" % (msg, tstr), flush=True)

#------------------------------------------------------------------------------
#
def reset_timer():
  init_timer()

#------------------------------------------------------------------------------
