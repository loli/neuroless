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
# since 2014-10-08
# status Development

# build-in module

# third-party modules
import numpy
from scipy.ndimage.morphology import binary_fill_holes
from medpy.io import load, header, save
from medpy.filter.binary import size_threshold

# own modules
from neuroless import FileSet, TaskMachine

# constants

# code
def postprocess(directory, inset, threshold):
    r"""
    Postprocess segmentation results.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    threshold : float
        All unconnected binary objects whose size in *ml* is smaller than this value are
        removed.
        
    Returns
    -------
    resultset : FileSet
        A FilSet centered on ``directory`` and representing the post-processed segmentations.
    """
    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
        
    # prepare output
    resultset = FileSet.fromfileset(directory, inset)

    # prepare and register skull-stripping tasks
    for case in inset.cases:
        src = inset.getfile(case=case)
        dest = resultset.getfile(case=case) 
        tm.register([src], [dest], _postprocess, [src, dest, threshold], dict(), 'post-processing')
        
    # run
    tm.run()        
            
    return resultset

def _postprocess(src, dest, threshold):
    r"""
    Execute post-processing on a segmentation.
    """
    # load source image
    img, hdr = load(src)
    img = img.astype(numpy.bool)
    
    # fill holes in 3D
    img = binary_fill_holes(img)
    # adapt threshold by voxel spacing
    threshold /= numpy.prod(header.get_pixel_spacing(hdr))
    # threshold binary objects
    out = size_threshold(img, threshold, 'lt')
    # reset if last object has been removed
    if 0 == numpy.count_nonzero(out):
        out = img
    # fill holes in 2d
    out = _fill2d(out)
    
    # save
    save(out, dest, hdr, True)
    
def _fill2d(arr, structure = None, dimension = 2):
    res = numpy.zeros(arr.shape, numpy.bool)
    for sl in range(arr.shape[dimension]):    
        res[:,:,sl] = binary_fill_holes(arr[:,:,sl], structure)
    return res
