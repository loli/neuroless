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
# since 2014-09-25
# status Development

# build-in module
import os

# third-party modules

# own modules
from ..ImageSet import ImageSet
from ..shell import call
from . import Action, Skullstrip
from ..exceptions import CommandExecutionError
from medpy.io import load, save
from neuroless.utilities import get_affine, set_qform, set_sform, set_qform_code,\
    set_sform_code

# constants

# code
class Biasfield (Action):
    r"""
    Computes and corrects the bias fields of MR images.
    """
    
    ACTIONDIR = '03biasfield'
    """The default action working directory name."""
    
    def __init__(self, cwd, imageset, brainmaskset):
        """
        Parameters
        ----------
        cwd : string
            The working directory in which to execute the action.        
        imageset : ImageSet
            The input image set.
        brainmaskset : ImageSet
            The image set containing the brain masks created with `Skullstrip`.
        """
        super(Biasfield, self).__init__(cwd)
        self.inset = imageset
        self.inset.validate()
        self.brainmaskset = brainmaskset
        self.brainmaskset.validate()
        
    @property
    def cawd(self):
        """
        Returns
        -------
        The current action working directory.
        """
        return os.path.join(self.cwd, Biasfield.ACTIONDIR)        
        
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
        self.outset = ImageSet.fromimageset(self.cawd, self.inset)
        
        # prepare the tasks
        self.tasks = []
        
        # bias-field correction
        for case, filename in self.inset.iterfilenames():
            src = self.inset.getfilebyfilename(case, filename)
            dest = self.outset.getfilebyfilename(case, filename)
            brainmaskfile = self.brainmaskset.getfilebysequence(case, Skullstrip.BRAINMASK_PSEUDO_SEQUENCE)
            self.tasks.append([[src, brainmaskfile], [dest], self.__correctbiasfield, [src, brainmaskfile, dest], dict(), 'bias-field'])
                
    def _postprocess(self):
        self.outset.validate()
        
    def __correctbiasfield(self, src, bmask, dest):
        """
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
        
        # correct the NIfTI header meta-data (it gets screwed up by cmtk)
        i, h = load(dest)
        aff = get_affine(h)
        set_qform(h, aff)
        set_sform(h, aff)
        set_qform_code(h, 1)
        set_sform_code(h, 1)
        save(i, dest, h, force=True)
