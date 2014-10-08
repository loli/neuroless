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
# version d0.1
# since 2014-10-06
# status Development
from neuroless import TaskMachine, FileSet
from medpy.features.utilities import join
import numpy
from medpy.io import load, save

# build-in module

# third-party modules

# own modules

# constants
PROBABILITY_THRESHOLD = 0.5
"""The probability of belonging to the foreground a voxel must reach to be considered object."""

# code
def applyforest(directory, forest, featureset, brainmasks):
    r"""
    Apply a forest to images to segment objects.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    forest : "ForestInstance"
        An instance of a trained forest from ``scikit.learn``.
    featureset : FileSet
        The features of the images.
    brainmasks : FileSet
        The associated brain masks.
            
    Returns
    -------
    segmentationset : FileSet
        A FileSet centered on ``directory`` and containing the segmentation masks.
    probabilityset : FileSet
        A FileSet centered on ``directory`` and containing the segmentation probabilities.
    """
    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
    
    # prepare output
    segmentationset = FileSet(directory, featureset.cases, None, ['{}_segmentation.nii.gz'.format(c) for c in featureset.cases], 'cases', False)
    probabilityset = FileSet(directory, featureset.cases, None, ['{}_probability.nii.gz'.format(c) for c in featureset.cases], 'cases', False)

    # register forest application tasks tasks
    for case in featureset.cases:
        featurefiles = featureset.getfiles(case=case)
        brainmaskfile = brainmasks.getfile(case=case)
        segmentationfile = segmentationset.getfile(case=case)
        probabilityfile = probabilityset.getfile(case=case)
        tm.register(featurefiles + [brainmaskfile],
                    [segmentationfile, probabilityfile],
                    __applyforest,
                    [forest, featurefiles, brainmaskfile, segmentationfile, probabilityfile],
                    dict(),
                    'forest-application')

    # run
    tm.run()  
            
    return segmentationset, probabilityset

def __applyforest(forest, featurefiles, brainmaskfile, segmentationfile, probabilityfile):
    """Apply a forest using the features and save the results."""
    # memory-efficient loading of the features for this case
    features = join(*[numpy.load(featurefile, mmap_mode='r') for featurefile in featurefiles])
    if 1 == features.ndim:
        features = numpy.expand_dims(features, -1)
    
    # apply forest
    probability_results = forest.predict_proba(features)[:,1]
    classification_results = probability_results > PROBABILITY_THRESHOLD # equivalent to forest.predict
    
    # create result image
    m, h = load(brainmaskfile)
    m = m.astype(numpy.bool)
    oc = numpy.zeros(m.shape, numpy.uint8)
    op = numpy.zeros(m.shape, numpy.float32)
    oc[m] = numpy.squeeze(classification_results).ravel()
    op[m] = numpy.squeeze(probability_results).ravel()

    # saving the results
    save(oc, segmentationfile, h)
    save(op, probabilityfile, h)
