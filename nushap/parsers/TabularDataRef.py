#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:58:46 2024

Reference parser for tabular data.

@author: jpms
"""

import sys
import os

import pandas as pd

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from utils import config

#------------------------------------------------------------------------------
#
def parse_tabular_data(fname):
  '''
    Parses tabular data. Cleans up the resulting data table.
  '''
  tab = pd.read_csv(fname)
  if config.colid:   # Remove input ID column, if declared
    tab = tab.drop(tab.columns[0], axis=1)
  # Convert dates to strings...
  dtype_dict = tab.dtypes.replace({'object': 'string'}).to_dict() #, 'datetime': 'string'
  tab = tab.astype(dtype_dict)
  if config.debug:
    ##print('Tab:\n', tab)
    print('DTypes:', tab.dtypes)
  if config.debug:
    print('Tmp tab:\n', tab)
  tab = tab.drop(config.excludes, axis=1)
  if config.drop_na_cols:
    tab = tab.loc[:, (tab.isnull().sum(axis=0) == 0)]  
  return tab

#------------------------------------------------------------------------------
#
def main():
  config.colid = False
  config.debug = True
  tab = None
  if len(sys.argv) == 2:
    tab = parse_tabular_data(sys.argv[1])
  else:
    tab = parse_tabular_data('../Examples/cars.csv')  # default example...
  print(tab)
  print('Table rows:')
  for index, row in tab.iterrows():
    print(tab.iloc[[index]])
    ##print(row)

#------------------------------------------------------------------------------
#
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------
  