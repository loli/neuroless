"""
======================================
Utilities (:mod:`neuroless.utilities`)
======================================
.. currentmodule:: neuroless.utilities

A number of small, internal utilities used by the NeuroLess package.

.. module:: neuroless.utilities
.. autosummary::
    :toctree: generated/
    
    set_qform_code
    set_sform_code
    set_qform
    set_sform
    get_qform_code
    get_sform_code
    get_affine
    get_qform
    get_sform
    get_diagonal
    
"""

# Copyright (C) 2013 Oskar Maier
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version r0.1
# since 2014-09-26
# status Development

# build-in module

# third-party modules
import numpy
from medpy.io import header

# own modules

# constants

# documentation templates

# code
def set_qform_code(h, v):
    h.get_header()['qform_code'] = v

def set_sform_code(h, v):
    h.get_header()['sform_code'] = v

def set_qform(h, v):
    h.set_qform(v)

def set_sform(h, v):
    h.set_sform(v)

def get_qform_code(h):
    return h.get_header()['qform_code']

def get_sform_code(h):
    return h.get_header()['sform_code']

def get_affine(h):
    return h.get_affine()

def get_qform(h):
    return h.get_qform()

def get_sform(h):
    return h.get_sform()

def get_diagonal(h):
    return numpy.diag(list(header.get_pixel_spacing(h)) + [1])