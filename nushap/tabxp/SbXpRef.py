#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 08:15:07 2024

Reference implementation of the computation of sample-based XPs.
Both the table and instance are required to be a dataframe.

ToDo:
* expand_axp_domains
* find_all_axp_dfs

@author: jpms
"""

import sys
import os

import numpy as np

root_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_name)

from utils import config
from utils import report

from parsers import TabularDataRef as td

#------------------------------------------------------------------------------
# Class that is used for computing Xps
#
class SbXp:

  #----------------------------------------------------------------------------
  #
  def __init__(self, tab, inst):
    self.has_wcxp = False
    self.has_cxp = False
    self.is_sorted = False
    self.wcxp = None
    self.cxps = None
    self.cxpset = None
    self.nf = None
    self.sc = None
    self.hits = None
    self.tab = tab
    self.inst = inst

  #----------------------------------------------------------------------------
  #
  def __str__(self):
    return 'WCXps:\n' + self.wcxp.__str__()

  #----------------------------------------------------------------------------
  #
  def sort_wcxp(self):
    # Obs: This adds a O(nlogn) component to the run time complexity
    #      Sorting is unnecessary for AXps, but helpful for CXps
    self.wcxp = self.wcxp.sort_values(by='_TCounts_')
    self.is_sorted = True

  #----------------------------------------------------------------------------
  #
  def check_no_overlap(self):
    chk0s = self.wcxp.any(bool_only=True, axis=1)
    ##print('chk0s:', chk0s, flush=True)
    if not chk0s.iloc[0]:
      report.prt_error('overlap -- matching rows with different targets...')

  #----------------------------------------------------------------------------
  #
  def construct_wcxp_matrix_ref(self):
    '''
      Constructs table of WCXps. The data table is assumed to respect format,
      e.g. no input ID column
      Running time is O(mn), with an extra O(nlogn) for sorting.
    '''
    if self.has_wcxp:
      return
    # 1. Copy reference table
    self.wcxp = self.tab.copy()     # New table, to contain the WCXps
    # 2. Get info on shape of WCXp matrix
    (nrow, ncol) = self.wcxp.shape
    predcol = config.target
    cols = list(self.wcxp.columns)    # List of original columns
    if predcol == None:
      predcol = cols[ncol-1]
    # 3. Add "special" ID column (after prediction column)
    self.wcxp[config.REFID] = range(nrow)
    # 4. Remove rows predicting the same class & then remove prediction column
    refval = self.inst.loc[0,predcol]
    self.wcxp = self.wcxp.drop(self.wcxp[(self.wcxp[predcol] == refval)].index)
    self.wcxp = self.wcxp.drop([predcol], axis=1)
    self.nf = ncol - 1
    # 5. Find matching/non-matching values
    ##cols.pop()
    cols.remove(predcol)
    for col in cols:
      refval = self.inst.loc[0,col]
      self.wcxp[col] = np.where(self.wcxp[col] == refval, False, True)
    # 6. Compute sums of differences
    self.wcxp['_TCounts_'] = self.wcxp[cols].sum(axis=1)
    # 7. Sort table of WCXps by size
    self.sort_wcxp()
    self.check_no_overlap()
    if config.debug:
      print('## WCXps:\n', self.wcxp, flush=True)
    if config.verbosity >= config.HIGHVERB:
      (nr, nc) = self.wcxp.shape
      print('## WCXps shape: (NR=%d, NC=%d)\n' % (nr, nc-2))
    # 9. Wrap-up
    self.has_wcxp = True

  #----------------------------------------------------------------------------
  #
  def construct_wcxp_matrix(self):
    self.construct_wcxp_matrix_ref()

  #----------------------------------------------------------------------------
  #
  def is_subset(self, rowA, rowB):
    (_, ncols) = self.wcxp.shape
    for idx in range(ncols-2):
      #print('Comparing: %d vs. %d' % (self.wcxp[i2,j], self.wcxp[i1,j]))
      if self.wcxp.iloc[rowB,idx] and not self.wcxp.iloc[rowA,idx]:
        #print('No subset')
        #print(self.wcxp[rowA,:], ' vs.\n ', self.wcxp[rowB, :])
        return False
    return True

  #----------------------------------------------------------------------------
  #
  def find_one_cxp(self):
    '''
      By default, returns the smallest CXp, i.e. the one in row 0...
      Algorithm runs in O(m) time.
    '''
    cxp = self.wcxp.columns[:-1][self.wcxp.iloc[0, :-1] == True].tolist()
    return cxp

  #----------------------------------------------------------------------------
  #
  def find_all_cxps_ref(self):  # Sequential mode
    '''
      Computes he (current) row numbers representing CXps [sequential version]
      The result updates cxpset
    '''
    ##if config.debug:
    ##  print('WCXps for CXps:\n', self.wcxp)
    # 1. Order rows by number of True entries...
    # Obs: this is already being done by default
    # 2. Compare all pairs of WCXps
    (nrows, ncols) = self.wcxp.shape
    self.cxpset = set()
    #trgt = set(list(range(nrows)))
    trgt = set(range(nrows))
    for i1 in range(nrows):
      if i1 not in trgt:
        continue
      trgt.remove(i1)
      keeps = set()
      for i2 in trgt:
        if not self.is_subset(i2, i1):
          keeps.add(i2)
      trgt = keeps
      self.cxpset.add(i1)
    self.has_cxp = True

  #----------------------------------------------------------------------------
  #
  def find_all_cxps(self):
    if self.has_cxp:
      return self.cxps
    self.find_all_cxps_ref()
    cxpvec = list(self.cxpset)
    cxpvec.sort()           # Obs: This bumps adds O(nlogn) to the running time
    #self.cxps = self.wcxp[cxpvec, :-2]
    self.cxps = self.wcxp.iloc[cxpvec, :-2]
    (_, ncols) = self.wcxp.shape # '_TCounts_'
    self.cxpcnts = self.wcxp.iloc[cxpvec , ncols-1 ].tolist()
    if config.debug:
      print('## CXps:\n', self.cxps)
      print('## Counts (%d): %s\n' % (len(self.cxpcnts), self.cxpcnts))
    return self.cxps

  #----------------------------------------------------------------------------
  #
  def extract_cxp_matrix_ref(self):
    (nrow, ncol) = self.cxps.shape
    ##matrix = [ [] ] * nrow
    mat = [[0 for _ in range(ncol)] for _ in range(nrow)]
    for row in range(nrow):
      ##matrix[row] = [0] * ncol
      for col in range(ncol):
        mat[row][col] = self.cxps.iloc[row,col]
    ##print('Matrix:', mat)
    return mat

  #----------------------------------------------------------------------------
  #
  def extract_cxp_matrix_np(self):
    return self.cxps.to_numpy()

  #----------------------------------------------------------------------------
  #
  def is_weak_cxp(self, wcxp):
    '''
      Decides whether set of column names (wcxp) is indeed a WCXp.
      Running time is O(|wcxp|n), which is O(mn).
      wcxp is expected to be a list of feature names.
      Obs: Flip order of row/col in the 1st loop and then collapse loops; no
           need for array of counts...
    '''
    (nrow, ncol) = self.wcxp.shape
    counts = [0] * nrow
    for coln in wcxp:
      cloc = self.wcxp.columns.get_loc(coln)
      for row in range(nrow):        
        if self.wcxp.iloc[row,cloc]:
          counts[row] += 1
    #print('Existing count at 0:', self.wcxp[0,'_TCounts_'])
    tccol = self.wcxp.columns.get_loc('_TCounts_')
    for row in range(nrow):
      #print('Existing counts:', self.wcxp[row,'_TCounts_'])
      if counts[row] == self.wcxp.iloc[row,tccol]:
        return True
    return False

  #----------------------------------------------------------------------------
  #
  def expand_axp_domains(self, axp):
    '''
      Implement Xp inflation
    '''
    print('Unavailable task -- domain expansion(s). Terminating...')
    sys.exit(-1)

  #----------------------------------------------------------------------------
  #
  def find_one_axp(self):
    '''
      Finds one AXp in O(mn) time, where m is the number of features and
      n is the number of samples in the dataset. The algorithm's complexity
      improves over known results, e.g. Cooper&Amgoud, ECAI'23.
      The insight is the use of counters, which are computed in O(mn) time.
      The sorting of rows in preprocessing is serve to reduce the running
      time in practice, but that is not being explored at the moment.
      OBS: use matrix of CXps if available; it should be fairly faster!!
    '''
    # ToDo: expand domains for non-categorical features
    # 1. Pick order of the features
    # ToDo ... starting with arbitrary (e.g. input) order
    # 2. Analyze each feature in order & decide whether feature can be dropped
    cols, counters, tcxps = None, None, None
    if not self.has_cxp:
      cols = list(self.wcxp.columns)      # Names of columns in the dataframe
      cntkey = cols.pop()                 # Key to access counters
      cols.pop()                          # Also pop refID
      counters = self.wcxp[cntkey].tolist()
      ##print('Counters:\n', counters)
      tcxps = self.wcxp
    else:
      cols = list(self.cxps.columns)
      counters = self.cxpcnts
      tcxps = self.cxps
    ##print('Counters:', counters)
    axp = set(cols)                     # Initial set of features in AXp
    ncols = len(cols)                   # Actual number of relevant columns
    nrow = len(counters)
    for j in range(ncols):
      keep = False
      for i in range(nrow):
        if tcxps.iloc[i,j]:
          counters[i] -= 1
          if counters[i] == 0:
            ##print('j: %d (%s) kept, because of %d' % (j, cols[j], i))
            keep = True
      if keep == True:
        for i in range(nrow):
          if tcxps.iloc[i,j]:
            counters[i] += 1
      else:
        axp.remove(cols[j])
    ##print('AXp:', axp)
    # 3. Inflate non-categorical features
    # ToDo ...
    return list(axp)

  #----------------------------------------------------------------------------
  #
  def create_mxsat_encoding(self):
    '''
      MaxSAT encoding for finding one smallest AXp.
      OBS: use matrix of CXps if available; it should be fairly faster!!
    '''
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
    if not self.has_cxp:
      self.find_all_cxps()
    (nrow, ncol) = self.cxps.shape    # shape of CXps contains only features
    mxsat = RC2(WCNF())               # passing an empty WCNF() formula
    for i in range(nrow):
      newcl = []
      for j in range(ncol):
        if self.cxps.iloc[i,j]:
          newcl.append(j+1)
      mxsat.add_clause(newcl)
      ##print('New Cl:', newcl)
    for j in range(ncol):
      mxsat.add_clause([-j-1], weight=1)
      ##print('New Cl:', [-j-1])
    return mxsat

  #----------------------------------------------------------------------------
  #
  def next_smallest_axp(self, mxsat):
    mxsat = self.create_mxsat_encoding()
    model = mxsat.compute()
    if config.debug:
      print('Cost: %d  ;  Model: %s' % (mxsat.cost, model))
    return model

  #----------------------------------------------------------------------------
  #
  def extract_smallest_axp(self, model):
    cols = list(self.wcxp.columns)[:-2]  # Set of original columns
    minaxp = []
    for vlit in model:
      if vlit > 0:
        col = cols[vlit-1]
        minaxp.append(col)
    return minaxp

  #----------------------------------------------------------------------------
  #
  def block_smallest_axp(self, mxsat, model):
    newcl = []
    for vlit in model:
      if vlit > 0:
        newcl.append(-vlit)
    mxsat.add_clause(newcl)
    if config.debug:
      print('New Blocking Cl:', newcl)

#----------------------------------------------------------------------------
  #
  def find_smallest_axp(self):
    '''
      Uses a MaxSAT solver for computing a smallest (minimum-cost)
      AXp. This is essentially a minimum set covering problem, that
      can be solved with other methods of automated reasoning.
    '''
    mxsat = self.create_mxsat_encoding()
    model = self.next_smallest_axp(mxsat)
    minaxp = self.extract_smallest_axp(model)
    mxsat.delete()
    if config.debug:
      print('Smallest AXp:', minaxp)
    return minaxp

  #----------------------------------------------------------------------------
  #
  def find_all_axp_sat(self):         # NOTE: implement the CAMUS algorithm
    '''
      SAT-based enumeration of AXps
    '''
    mxsat = self.create_mxsat_encoding()
    min_axps = []
    num_axps = 0
    for model in mxsat.enumerate(block=1):
      minaxp = self.extract_smallest_axp(model)
      min_axps.append(minaxp)        
      if config.debug:
        print('Model:', model)
        print('Smallest AXp:', minaxp)
      self.block_smallest_axp(mxsat, model)
      num_axps += 1
      if num_axps >= config.axp_enum_budget and not config.axp_full_enum:
        break
    return min_axps        

  #----------------------------------------------------------------------------
  #
  def find_all_axp_dfs(self):         # NOTE: implement the CAMUS algorithm
    '''
      DFS-based enumeration of AXps (planned)
    '''
    print('Unavailable task -- listing all AXps, dfs mode. Terminating...')
    sys.exit(-1)

  #----------------------------------------------------------------------------
  #
  def find_all_axp(self):         # NOTE: implement the CAMUS algorithm
    '''
      Lists all AXps, but a budget can be specified. One solution
      exploits a SAT oracle. The other (planned) solution uses a
      CAMUS-like (i.e. DFS) approach.
    '''
    if config.axp_enum_mode == 'sat':
      return self.find_all_axp_sat()
    return self.find_all_axp_dfs()

  #----------------------------------------------------------------------------
  #
  def init_check_waxp(self):
    wcxpf = self.wcxp if not self.has_cxp else self.cxps  # CXp frame to use
    (nrow, _) = wcxpf.shape    # Could use (all) cxp instead
    self.hits = [False] * nrow

  #----------------------------------------------------------------------------
  #
  def is_weak_axp(self, waxp):
    '''
      Decides whether a set of column names (waxp) is indeed a WAXp. 
      The run time is O(|waxp|n), which is O(mn).
      waxp is expected to be a list of feature names.
      CXpMat proposes a more efficient solution, used elsewhere.
    '''
    if self.hits == None:
      self.init_check_waxp()
    wcxpf = self.wcxp if not self.has_cxp else self.cxps  # CXp frame to use
    (nrow, _) = wcxpf.shape    # Could use (all) cxp instead
    rv = True
    for row in range(nrow):
      if self.hits[row]:
        self.hits[row] = False
        continue
      for coln in waxp:
        colidx = wcxpf.columns.get_loc(coln)
        if wcxpf.iloc[row,colidx]:
          self.hits[row] = True
          break
      if not self.hits[row]:
        rv = False
      else:
        self.hits[row] = False
    return rv

  #----------------------------------------------------------------------------
  #
  def list_all_nirf(self):
    '''
      Finds cxp-necessary (& axp-necessary) features, as well as
      relevant features.
      Running time is O(mn^2) (in sequential mode), under the assumption
      that the CXps need to be computed.
      cxps denotes the list of rows representing CXps.
    '''
    if not self.has_cxp:
      self.find_all_cxps()
    (nrow, ncol) = self.wcxp.shape
    # 1. Compute the set of relevant features
    collst = list(self.wcxp.columns)[:-2]        # Set of original columns    
    ##print('Goal cols:', collst)
    ##print('CXp set:', self.cxpset)
    colset = set(collst)
    rfset = set()
    for row in self.cxpset:
      dropset = set()
      for col in colset:
        colidx = self.wcxp.columns.get_loc(col)
        if self.wcxp.iloc[row,colidx]:
          dropset.add(col)
      for dcol in dropset:
        colset.remove(dcol)
        rfset.add(dcol)
    rfs = list(rfset)
    # 2. Compute the AXp-necessary features
    anfs = []
    for row in self.cxpset:
      cntidx = self.wcxp.columns.get_loc('_TCounts_')
      if self.wcxp.iloc[row,cntidx] == 1:# Feature must be hit => AXp-necessary
        for col in collst:
          colidx = self.wcxp.columns.get_loc(col)  
          if self.wcxp.iloc[row,colidx]:
            anfs.append(col)
    # 3. Compute the CXp-necessary features
    gnrow = len(self.cxpset)
    cnfs = []
    for col in collst:
      colidx = self.wcxp.columns.get_loc(col)
      sum = 0
      for row in self.cxpset:
        if self.wcxp.iloc[row,colidx]:
          sum += 1
      if sum == gnrow:                   # Feature must be hit => CXp-necessary
        cnfs.append(col)
      ##else:
      ##  print('Feat. ', col, ' is not CXp-necessary')
    return anfs, cnfs, rfs

#------------------------------------------------------------------------------
# Used for testing purposes
#
def main():
  config.colid = True
  tab = td.parse_tabular_data('../Tests/han-dmct12-tab81.csv')
  #tab = td.parse_tabular_data('../Examples/Loan.csv')
  print('Dataset:\n', tab)
  inst = td.parse_tabular_data('../Tests/han-dmct12-tab81-inst.csv')
  #inst = td.parse_tabular_data('../Examples/Loan-inst.csv')
  print('Instance:\n', inst)
  sbxp = SbXp(tab, inst)
  sbxp.construct_wcxp_matrix_seq()
  print('SbXp object:\n', sbxp)
  axp = sbxp.find_one_axp()
  print('AXp:', axp)
  cxps = sbxp.construct_cxp_matrix_seq()
  print('CXps:', cxps)
  
#------------------------------------------------------------------------------
#
if __name__ == '__main__':
  main()
