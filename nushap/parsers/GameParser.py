#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:32:53 2024

Module implementing the parsing of weighted voting games (WVGs).

@author: jpms
"""

import sys
import os

import re

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from games import WVG
from games import DWVG

#------------------------------------------------------------------------------
#
def parse_game(strdef, adddelay=False):
  '''
    Parser for weighted voting games.
  '''
  expr = re.compile('\[\s*(\d+)\s*;\s*([\d\s,]+)\s*\]')
  qval, wstr = expr.search(strdef).groups()
  qval = int(qval)
  wvstr = list(map(str.strip, wstr.split(',')))
  wvec = list(map(int, wvstr))
  game = WVG.WVG(int(qval), wvec) if not adddelay else DWVG.DWVG(int(qval), wvec)
  return game

#------------------------------------------------------------------------------
#
if __name__ == "__main__":
  '''
    The main serves for testing purposes
  '''
  if len(sys.argv) != 2:
    print("Usage: GameParser.py <game-def>", file=sys.stderr)
    sys.exit(-1)
  wvg = parse_game(sys.argv[1])
  print("WVG: ", wvg)

#------------------------------------------------------------------------------
