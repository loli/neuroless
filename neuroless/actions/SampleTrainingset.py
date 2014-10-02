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
# since 2014-09-26
# status Development

# build-in module
import os

# third-party modules
import numpy
from medpy.features.utilities import append

# own modules
from ..ImageSet import ImageSet
from . import Action, Skullstrip


# constants

# code
class SampleTrainingset (Action):
    r"""
    Sample a training set from the features.
    """
    
    ACTIONDIR = '06smaplingset'
    """The default action working directory name."""
    TRAININGSET_PSEUDO_SEQUENCE = 'trainset'
    """A pseudo sequence name for the sampled training set."""
    
    def __init__(self, cwd, imageset):
        """
        Parameters
        ----------
        cwd : string
            The working directory in which to execute the action.
        imageset : ImageSet
            The input image set, here the feature set.
        """
        super(SampleTrainingset, self).__init__(cwd)
        self.inset = imageset
        self.inset.validate()
        
    @property
    def cawd(self):
        """
        Returns
        -------
        The current action working directory.
        """
        return os.path.join(self.cwd, SampleTrainingset.ACTIONDIR)
        
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
        self.outset = ImageSet.fromimageset(self.cawd, [''], [SampleTrainingset.TRAININGSET_PSEUDO_SEQUENCE])
        
        # prepare the tasks
        self.tasks = []
        
        # feature sampling
        featurefiles = [f for _, f in self.inset.iterfiles()]
        trainset = self.outset.getfilebysequence('', SampleTrainingset.TRAININGSET_PSEUDO_SEQUENCE)
        self.tasks.append([featurefiles], [trainset], self.__sample, [trainset] + featurefiles, dict(), 'sample-trainingset')
        
    def _postprocess(self):
        self.outset.validate()
        
    def __sample(self, trainset, *featurefiles):
        """
        Extract a training sample set from the supplied feature sets.
        
        Parameters
        ----------
        trainset : strings
            The target training set file.
        *featurefiles : strings
            The features used for training.
        """
        # memory-efficient loading of the features
        full_feature_set = numpy.load(featurefiles[0], mmap_mode='r') 
        for featurefile in featurefiles[1:]:
            full_feature_set = append(full_feature_set, numpy.load(featurefile, mmap_mode='r'))
            
            
