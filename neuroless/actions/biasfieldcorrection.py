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
# since 2014-10-02
# status Development

# build-in module
import itertools
import os

# third-party modules
from medpy.io import save, load

# own modules
from .. import FileSet, TaskMachine
from ..shell import call
from ..exceptions import CommandExecutionError
from ..utilities import get_affine, set_qform, set_sform, set_qform_code,\
    set_sform_code

# constants

# code
def correctbiasfields(directory, inset, brainmasks):
    r"""
    Compute and correct the bias fields of MR images.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    brainmasks : FileSet
        The associated brain masks file set.
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and containing the bias field corrected
        images.
    """
    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
    
    # prepare output
    resultset = FileSet.fromfileset(directory, inset)
        
    # register bias-field correction tasks
    for case, identifier in itertools.product(inset.cases, inset.identifiers):
        src = inset.getfile(case=case, identifier=identifier)
        dest = resultset.getfile(case=case, identifier=identifier)
        brainmaskfile = brainmasks.getfile(case=case)
        tm.register([src, brainmaskfile], [dest], _correctbiasfield, [src, dest, brainmaskfile], dict(), 'bias-field')
                
    # run
    tm.run()
                
    return resultset
        
def _correctbiasfield(src, dest, bmask):
    r"""
    Correct the bias field of an image.
    
    Parameters
    ----------
    src : string
        Path to the image to correct.
    dest : string
        Target location for the bias-field corrected image.
    bmask : string
        A binary image where the non-zero values denote the area over which to
        compute the bias field.
    """
    # prepare and run skull-stripping command
    cmd = ['cmtk', 'mrbias', '--mask', bmask, src, dest]
    rtcode, stdout, stderr = call(cmd)
    
    # check if successful
    if not os.path.isfile(dest):
        raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Bias-corrected image not created.')
    
    _correctniftiheader(dest)
    
def _correctniftiheader(image):
    r"""
    Correct the NIfTI header meta-data of a file in-place.
    This is usually required after an application of CMTK, as this screwes up the header.
    """
    # correct the NIfTI header meta-data (it gets screwed up by cmtk)
    i, h = load(image)
    aff = get_affine(h)
    set_qform(h, aff)
    set_sform(h, aff)
    set_qform_code(h, 1)
    set_sform_code(h, 1)
    save(i, image, h, force=True)
