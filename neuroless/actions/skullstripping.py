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
import os

# third-party modules
from medpy.core.logger import Logger

# own modules
from .. import FileSet, TaskMachine
from ..shell import call, mv
from ..exceptions import CommandExecutionError

# constants
SEQUENCE_PREFERENCES = ['t1', 't2', 'flair']
PREFERRED_FILE_SUFFIX = 'nii.gz'

# code
def stripskull(directory, inset, stripsequence = False):
    r"""
    Compute a brain mask for MR brain images.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    stripsequence : False or string
        The sequence to use for computing the brain mask. If none supplied, the order
        in `~skullstripping.SEQUENCE_PREFERENCES` is respected.
        
    Returns
    -------
    resultset : FileSet
        A FilSet centered on ``directory`` and representing the binary brain masks.
    stripsequence : string
        The sequence employed for the skull-stripping.
    """
    logger = Logger.getInstance()
    
    # decide on strip-sequence
    if not stripsequence:
        for sequence in SEQUENCE_PREFERENCES:
            if sequence in inset.identifiers:
                stripsequence = sequence
                break
        if not stripsequence:
            stripsequence = inset.identifiers[0]
            logger.warning('None of the preferred sequences for skull-stripping "{}" available. Falling back to "{}"'.format(SEQUENCE_PREFERENCES, stripsequence))
    elif not stripsequence in inset.identifiers:
        raise ValueError('The chosen skull-strip sequence "{}" is not available in the input image set.'.format(stripsequence))

    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
        
    # prepare output
    resultset = FileSet(directory, inset.cases, False, ['{}.{}'.format(cid, PREFERRED_FILE_SUFFIX) for cid in inset.cases], 'cases', False)

    # prepare and register skull-stripping tasks
    for case in inset.cases:
        src = inset.getfile(case=case, identifier=stripsequence)
        dest = resultset.getfile(case=case)
        rfile = dest.replace('.{}'.format(PREFERRED_FILE_SUFFIX),  '_mask.{}'.format(PREFERRED_FILE_SUFFIX)) 
        tm.register([src], [dest], brainmask, [src, dest, rfile], dict(), 'skull-strip')
        
    # run
    tm.run()        
            
    return resultset, stripsequence
        
def brainmask(src, dest, resultfile):
    r"""
    Computes a brain mask.
    
    Parameters
    ----------
    src : string
        Path to the image on which to compute the brain mask.
    dest : string
        Target location for the brain mask.
    resultfile : string
        The actual result file created by the external call.
    """
    # prepare and run skull-stripping command
    cmd = ['fsl5.0-bet', src, dest, '-n', '-m', '-R']
    rtcode, stdout, stderr = call(cmd)
    
    # check if successful
    if not os.path.isfile(resultfile):
            raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Brain mask image not created.')
        
    # copy
    mv(resultfile, dest)
