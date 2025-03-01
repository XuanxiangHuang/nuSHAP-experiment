#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:17:31 2024

@author: jpms
"""

import sys
import signal

from utils import times

#------------------------------------------------------------------------------
#
def sigint_handler(signum, frame):
  times.report_elapsed_time('Ctrl-c caught')
  res = input('Do you want to exit? y/n ')
  if res == 'y':
    sys.exit(1)
 
#------------------------------------------------------------------------------
#
def set_interrupt_handler():
  signal.signal(signal.SIGINT, sigint_handler)
  
#------------------------------------------------------------------------------
