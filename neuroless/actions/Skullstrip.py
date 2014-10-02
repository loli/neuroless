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
# since 2014-09-22
# status Development

# build-in module
import os

# third-party modules

# own modules
from . import Action
from ..ImageSet import ImageSet
from ..shell import call, mv
from neuroless.exceptions import CommandExecutionError

# constants

# code
class Skullstrip (Action):
    r"""
    Creates a brain mask for Mr brian images.
    """
    
    SEQUENCE_PREFERENCES = ['t1', 't2', 'flair']
    """Preferred sequences to use for skull-stripping in order."""
    ACTIONDIR = '02brainmasks'
    """The default action working directory name."""
    BRAINMASK_PSEUDO_SEQUENCE = 'brainmask'
    """A pseudo sequence name for the brainmask files."""
    
    def __init__(self, cwd, imageset, stripsequence = False):
        """
        Parameters
        ----------
        cwd : string
            The working directory in which to execute the action.        
        imageset : ImageSet
            The input image set.
        stripsequence : False or string
            The sequence to use for computing the brain mask. If none supplied, the order
            in `~Skullstrip.SEQUENCE_PREFERENCES` is respected.
        """
        super(Skullstrip, self).__init__(cwd)
        self.inset = imageset
        self.inset.validate()
        if not stripsequence:
            for sequence in Skullstrip.SEQUENCE_PREFERENCES:
                if sequence in self.inset.sequences:
                    stripsequence = sequence
            if not stripsequence:
                stripsequence = Skullstrip.SEQUENCE_PREFERENCES[0]
                self.logger.warning('None of the preferred sequences for skull-stripping "{}" available. Falling back to "{}"'.format(Skullstrip.SEQUENCE_PREFERENCES, stripsequence))
        elif not stripsequence in self.inset.sequences:
            raise ValueError('The chose skull-strip sequence "{}" is not available in the input image set.'.format(stripsequence))
        self.stripsequence = stripsequence
        
    @property
    def cawd(self):
        """
        Returns
        -------
        The current action working directory.
        """
        return os.path.join(self.cwd, Skullstrip.ACTIONDIR)
        
    def getimageset(self):
        """
        Returns
        -------
        outset : ImageSet
            The output image set.
        """
        return self.outset
        
    def _preprocess(self):
        # prepare output
        self.outset = ImageSet(self.cawd, self.inset.cases, ['brainmask'], ['{}.{}'.format(self.stripsequence, Action.PREFERRED_FILE_SUFFIX)])
        
        # prepare the tasks
        self.tasks = []

        # skull-stripping
        for case, filename in self.inset.iterfilenames(self.stripsequence):
            src = self.inset.getfilebyfilename(case, filename)
            trg = self.outset.getfilebysequence(case, Skullstrip.BRAINMASK_PSEUDO_SEQUENCE)
            rfile = trg.replace('.{}'.format(Action.PREFERRED_FILE_SUFFIX),  '_mask.{}'.format(Action.PREFERRED_FILE_SUFFIX)) 
            self.tasks.append([[src], [trg], self.__strip, [src, trg, rfile], dict(), 'skull-strip'])
                
    def _postprocess(self):
        self.outset.validate()
        
    def __strip(self, src, dest, resultfile):
        """
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

        