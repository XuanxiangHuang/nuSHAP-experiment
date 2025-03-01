#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:25:22 2024

String utilities.

@author: jpms
"""

#------------------------------------------------------------------------------
#
def fmtjoin(iterable, joiner=", ", format=""):
    format=f"{{{format}}}"
    return '[ ' + joiner.join(format.format(i) for i in iterable) + ' ]'

#------------------------------------------------------------------------------
