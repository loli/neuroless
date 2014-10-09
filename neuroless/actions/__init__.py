"""
===========================================
Pipeline actions (:mod:`neuroless.actions`)
===========================================
.. currentmodule:: neuroless.actions

This package contains various actions that can be used to build up a pipeline for
neuroscientific image processing. Most of the actions make use of the ``TaskMachine``,
and thus allow for parallel batch execution.

Unify and re-sample images :mod:`neuroless.actions.unification`
===============================================================

.. module:: neuroless.actions.unification
.. autosummary::
    :toctree: generated/
    
    unify
    resample
    resamplebyexample

Skull-stripping :mod:`neuroless.actions.skullstripping`
=======================================================

.. module:: neuroless.actions.skullstripping
.. autosummary::
    :toctree: generated/
    
    stripskull

Bias-filed correction :mod:`neuroless.actions.biasfieldcorrection`
==================================================================

.. module:: neuroless.actions.biasfieldcorrection
.. autosummary::
    :toctree: generated/
    
    correctbiasfields

Image intensity range standardisation :mod:`neuroless.actions.intensityrangestandardisation`
============================================================================================

.. module:: neuroless.actions.intensityrangestandardisation
.. autosummary::
    :toctree: generated/
    
    percentilemodelstandardisation
    percentilemodelapplication


Feature extraction and sampling :mod:`neuroless.actions.features`
=================================================================

.. module:: neuroless.actions.features
.. autosummary::
    :toctree: generated/
    
    extractfeatures
    sample
    stratifiedrandomsampling


Decision forest training :mod:`neuroless.actions.training`
==========================================================

.. module:: neuroless.actions.training
.. autosummary::
    :toctree: generated/
    
    trainet

Decision forest application :mod:`neuroless.actions.application`
================================================================

.. module:: neuroless.actions.application
.. autosummary::
    :toctree: generated/
    
    applyforest
    
Post-process segmentation results :mod:`neuroless.actions.postprocessing`
=========================================================================
 
.. module:: neuroless.actions.postprocessing
.. autosummary::
    :toctree: generated/
    
    postprocess

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

# if __all__ is not set, only the following, explicit import statements are executed
from unification import unify, resample, resamplebyexample
from skullstripping import stripskull
from biasfieldcorrection import correctbiasfields
from intensityrangestandardisation import percentilemodelstandardisation, percentilemodelapplication
from features import extractfeatures, sample
from training import trainet
from application import applyforest
from postprocessing import postprocess

# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]


