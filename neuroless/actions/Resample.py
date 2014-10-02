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
from medpy.filter.image import resample
import medpy.io

# own modules
from . import Action
from ..ImageSet import ImageSet

# constants

# code
class Resample (Action):
    r"""
    Re-samples images to a common space.
    """
    
    BSPLINE_ORDER = 3
    """The order of the b-spline-interpolation employed during the re-sampling."""
    ACTIONDIR = '00resampledgt'
    """The default action working directory name."""
    
    def __init__(self, cwd, groundtruthset, targetspacing = 1):
        """
        Parameters
        ----------
        cwd : string
            The working directory in which to execute the action.
        groundtruthset : ImageSet
            The ground truth image set.
        targetspacing : False or number or sequence of numbers
            The target spacing for all images. If ``False``, the original spacing of the
            ``fixedsequence`` image is kept; if a single number, isotropic spacing is
            assumed; a sequence of numbers denotes custom spacing.
        """
        super(Resample, self).__init__(cwd)
        self.inset = groundtruthset
        self.inset.validate()
        self.targetspacing = targetspacing # False = no re-sampling, single number = isotropic, sequence of numbers = custom
        
    @property
    def cawd(self):
        """
        Returns
        -------
        The current action working directory.
        """
        return os.path.join(self.cwd, Resample.ACTIONDIR)
        
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

        # Re-sampling respectively copy the fixed-sequence files
        for case, filename in self.inset.iterfilenames():
            src = self.inset.getfilebyfilename(case, filename)
            trg = self.outset.getfilebyfilename(case, filename)
            if self.targetspacing: # re-sample
                self.tasks.append([[src], [trg], self.__resample, [src, trg], dict(), 're-sample'])
            else: # simply copy
                self.tasks.append([[src], [trg], self.__scp, [src, trg], dict(), 'save-copy'])   

    def _postprocess(self):
        self.outset.validate()

    def __resample(self, src, trg):
        img, hdr = medpy.io.load(src)
        img, hdr = resample(img, hdr, self.targetspacing, Resample.BSPLINE_ORDER)
        medpy.io.save(img, trg, hdr)
    