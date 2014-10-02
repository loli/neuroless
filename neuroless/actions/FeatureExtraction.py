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
import pickle

# third-party modules
import numpy
from medpy.io import load, header
from medpy.features.utilities import join
from medpy.features.intensity import intensities, local_mean_gauss,\
    local_histogram, centerdistance_xdminus1

# own modules
from ..ImageSet import ImageSet
from . import Action, Skullstrip


# constants

# code
class FeatureExtraction (Action):
    r"""
    Extract features from the images.
    """
    
    ACTIONDIR = '05features'
    """The default action working directory name."""
    FEATURE_DTYPE = numpy.float32
    """The target dtype of all features."""
    CLASSMEMBERSHIP_PSEUDO_SEQUENCE = 'classes'
    """A pseudo sequence name for the class membership set."""
    
    def __init__(self, cwd, imageset, brainmaskset, groundtruthset):
        """
        Parameters
        ----------
        cwd : string
            The working directory in which to execute the action.        
        imageset : ImageSet
            The input image set.
        brainmaskset : ImageSet
            The image set containing the brain masks created with `Skullstrip`.
        groundtruthset : ImageSet
            The image set containing the ground-truth segmentations.
        """
        super(FeatureExtraction, self).__init__(cwd)
        self.inset = imageset
        self.inset.validate()
        self.brainmaskset = brainmaskset
        self.brainmaskset.validate()
        self.groundtruthset = groundtruthset
        self.groundtruthset.validate()
        
    @property
    def cawd(self):
        """
        Returns
        -------
        The current action working directory.
        """
        return os.path.join(self.cwd, FeatureExtraction.ACTIONDIR)        
        
    def getimageset(self):
        """
        Returns
        -------
        outset : ImageSet
            The output image set.
        """
        return self.outset
    
    def getfeaturenameset(self):
        """
        Returns
        -------
        featurenameset : ImageSet
            The feature name set
        """
        return self.featurenameset
    
    def getclassmembershipset(self):
        """
        Returns
        -------
        classmembershipset : ImageSet
            The class memberships per case.
        """
        return self.classmembershipset    
        
    def _preprocess(self):
        # prepare filenames for feature matrices
        filenames = []
        fnfilenames = []
        for filename in self.inset.files:
            if filename.endswith(Action.PREFERRED_FILE_SUFFIX):
                fnfilename = filename[:-len(Action.PREFERRED_FILE_SUFFIX)] + 'fnames.pkl'
                filename = filename[:-len(Action.PREFERRED_FILE_SUFFIX)] + 'npy'
            elif filename.find('.') > 0:
                fnfilename = filename[:filename.rfind('.')+1] + 'fnames.pkl'
                filename = filename[:filename.rfind('.')+1] + 'npy'
            else:
                fnfilename = filename + '.fnames.pkl'
                filename += '.npy'
            filenames.append(filename)
            fnfilenames.append(fnfilename)
        
        # prepare output        
        self.outset = ImageSet(self.cawd, self.inset.cases, self.inset.sequences, filenames)
        self.featurenameset = ImageSet(self.cawd, self.inset.cases, self.inset.sequences, fnfilenames)
        self.classmembershipset = ImageSet(self.cawd, self.inset.cases, [FeatureExtraction.CLASSMEMBERSHIP_PSEUDO_SEQUENCE], ['classmemberships.npy'])
        
        # prepare the tasks
        self.tasks = []
        
        # model training & model application
        for sequence in self.inset.sequences:
            for case, imagename in self.inset.iterfilenames([sequence]):
                imagefile = self.inset.getfilebyfilename(case, imagename)
                brainmaskfile = self.brainmaskset.getfilebysequence(case, Skullstrip.BRAINMASK_PSEUDO_SEQUENCE)
                groundtruthfile = self.groundtruthset.getfilebysequence(case, 'gt') # !TODO: No very clean like this!
                destfile = self.outset.getfilebysequence(case, sequence)
                fndestfile = self.featurenameset.getfilebysequence(case, sequence)
                cmdestfile = self.classmembershipset.getfilebysequence(case, FeatureExtraction.CLASSMEMBERSHIP_PSEUDO_SEQUENCE)
                
                self.tasks.append([[imagefile, brainmaskfile, groundtruthfile],
                                   [destfile, cmdestfile, fndestfile],
                                   self.__extract,
                                   [imagefile, brainmaskfile, groundtruthfile, destfile, cmdestfile, fndestfile],
                                   dict(),
                                   'feature-extraction'])
                
    def _postprocess(self):
        self.outset.validate()
        self.featurenameset.validate()
        
    def __extract(self, imagefile, brainmaskfile, groundtruthfile, destfile, cmdestfile, fndestfile):
        """
        Extract all features from the supplied image.
        
        Parameters
        ----------
        imagefile : string
            The image from which to extract the features.
        brainmaskfile : string
            The corresponding brain mask.
        groundtruthfile : string
            The corresponding ground-truth file.
        destfile : string
            The destination file in which to save the computed feature vector.
        cmdestfile : string
            The destination file for the class memberships.
        fndestfile : string
            The deatination file for the feature names.
        """
        # loading the images
        img, hdr = load(imagefile)
        msk = load(brainmaskfile)[0].astype(numpy.bool)
        gt = load(groundtruthfile)[0].astype(numpy.bool)
        
        # prepare feature vector and the feature identification list
        feature_vector = None
        feature_names = []
        
        # iterate the features to extract
        for function_call, function_arguments, voxelspacing in FeatureExtraction.FEATURE_CONFIG:
            # extract the feature
            call_arguments = list(function_arguments)
            if voxelspacing: call_arguments.append(header.get_pixel_spacing(hdr))
            call_arguments.append(msk)
            print function_call.__name__, call_arguments[:-1]
            fv = function_call(img, *call_arguments)
            # append to the images feature vector
            if feature_vector is None:
                feature_vector = fv
            else:
                feature_vector = join(feature_vector, fv)
            # create and save feature names
            feature_name = '{}.{}'.format(function_call.__name__, '_'.join(map(str, function_arguments)))
            if fv.ndim > 1:
                feature_names.extend(['{}.{}'.format(feature_name, i) for i in range(fv.shape[0])])
            else:
                feature_names.append(feature_name)
        
        # save the extracted feature vector and the feature names
        with open(destfile, 'wb') as f:
            numpy.save(f, feature_vector.astype(FeatureExtraction.FEATURE_DTYPE))
        with open(fndestfile, 'wb') as f:
            pickle.dump(feature_names, f)
            
        # save the class memberships (truncated by the brain mask)
        with open(cmdestfile, 'wb') as f:
            pickle.dump(gt[msk], f)


    FEATURE_CONFIG = [
        (intensities, [], False),
        (local_mean_gauss, [3], True),
        (local_mean_gauss, [5], True),
        (local_mean_gauss, [7], True),
        (local_histogram, [11, 'image', (0, 100), 5, None, None, 'ignore', 0], False), #11 bins, 5*2=10mm region
        (local_histogram, [11, 'image', (0, 100), 10, None, None, 'ignore', 0], False), #11 bins, 10*2=20mm region
        (local_histogram, [11, 'image', (0, 100), 15, None, None, 'ignore', 0], False), #11 bins, 15*2=30mm region
        (centerdistance_xdminus1, [0], True),
        (centerdistance_xdminus1, [1], True),
        (centerdistance_xdminus1, [2], True)
    ]
    """The features to extract."""
