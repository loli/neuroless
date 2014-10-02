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
import pickle
import os

# third-party modules
from medpy.filter import IntensityRangeStandardization
from medpy.io import load, save
import numpy

# own modules
from ..ImageSet import ImageSet
from . import Action, Skullstrip

# constants

# code
class IntensityStandardisation (Action):
    r"""
    Train and apply intensity standardisation models for each sequence.
    """
    
    ACTIONDIR = '04intensitystd'
    """The default action working directory name."""
    MODELS_PSEUDO_CASENAME = 'trainedmodels'
    """A pseudo case name for the trained models."""
    
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
        super(IntensityStandardisation, self).__init__(cwd)
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
        return os.path.join(self.cwd, IntensityStandardisation.ACTIONDIR)        
        
    def getimageset(self):
        """
        Returns
        -------
        outset : ImageSet
            The output image set.
        """
        return self.outset
    
    def getmodelset(self):
        """
        Returns
        -------
        modelset : ImageSet
            The model set.
        """
        return self.modelset
        
    def _preprocess(self):
        # prepare output
        self.outset = ImageSet.fromimageset(self.cawd, self.inset)
        self.modelset = ImageSet(self.cawd, [IntensityStandardisation.MODELS_PSEUDO_CASENAME], self.inset.sequences, ['{}.{}'.format(s, Action.PREFERRED_FILE_SUFFIX) for s in self.inset.sequences])
        
        # prepare the tasks
        self.tasks = []
        
        # model training & model application
        for sequence in self.inset.sequences:
            trainingfiles = [f for _, f in self.inset.iterfiles([sequence])]
            brainmaskfiles = [f for _, f in self.brainmaskset.iterfiles([Skullstrip.BRAINMASK_PSEUDO_SEQUENCE])]
            destfiles = [f for _, f in self.outset.iterfiles([sequence])]
            destmodel = self.modelset.getfilebysequence(IntensityStandardisation.MODELS_PSEUDO_CASENAME, sequence)
            self.tasks.append([trainingfiles + brainmaskfiles, [destmodel] + destfiles,
                               self.__intensitystd,
                               [trainingfiles, brainmaskfiles, destfiles, destmodel],
                               dict(),
                               'intensity-standardisation'])
                
    def _postprocess(self):
        self.outset.validate()
        self.modelset.validate()
        
    def __intensitystd(self, trainingfiles, brainmaskfiles, destfiles, destmodel):
        """
        Train an intensity standardisation model and apply it. All values outside of the
        brain mask are set to zero.
        
        Parameters
        ----------
        trainingfiles : sequence of strings
            All images to use for training and to which subsequently apply the trained model.
        brainmaskfiles : sequence of strings
            The brain masks corresponding to ``trainingfiles``.
        destfiles : sequence of strings
            The intensity standarised target locations corresponding to ``trainingfiles``.
        destmodel : string
            The target model location.
        """
        # check arguments
        if not len(trainingfiles) == len(brainmaskfiles):
            raise ValueError('The number of supplied trainingfiles must be equal to the number of brainmaskfiles.')
        elif not len(trainingfiles) == len(destfiles):
            raise ValueError('The number of supplied trainingfiles must be equal to the number of destfiles.')
        
        # loading input images (as image, header pairs)
        images = []
        headers = []
        for image_name in trainingfiles:
            i, h = load(image_name)
            images.append(i)
            headers.append(h)
            
        # loading brainmasks
        masks = [load(mask_name)[0].astype(numpy.bool) for mask_name in brainmaskfiles]
            
        # train the model
        irs = IntensityRangeStandardization()
        trained_model, transformed_images = irs.train_transform([i[m] for i, m in zip(images, masks)])
        
        # condense outliers in the image (extreme peak values at both end-points of the histogram)
        transformed_images = [self.__condense(i) for i in transformed_images]
        
        # saving the model
        with open(destmodel, 'wb') as f:
            pickle.dump(trained_model, f)
        
        # save the transformed images
        for ti, i, m, h, dest in zip(transformed_images, images, masks, headers, destfiles):
            i[~m] = 0
            i[m] = ti
            save(i, dest, h)

    def __condense(self, img):
        """
        Apply a percentile threshold to the image, condensing all outliers to the percentile values.
        
        Parameters
        ----------
        img : array_like
            The image whose outliers to condense.
            
        Returns
        -------
        out : ndarray
            The resulting image.
        """
        out = img.copy()
        li = numpy.percentile(out, (1, 99.9))
        out[out < li[0]] = li[0]
        out[out > li[1]] = li[1]
        return out
