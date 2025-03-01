#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:16:52 2024

Basic numeric utilities.

@author: jpms
"""

#------------------------------------------------------------------------------
#
def num_dec_places(num):
  count = 0
  while (num < 1):
    num *= 10
    count += 1
  return count

#------------------------------------------------------------------------------
# Used for testing purposes
#
def main():
  print('Num decimal places:')
  print(num_dec_places(0.0005))
  print(num_dec_places(0.001))
  print(num_dec_places(0.05))

#------------------------------------------------------------------------------
#
if __name__ == '__main__':
  main()

#------------------------------------------------------------------------------
