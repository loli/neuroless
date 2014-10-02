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
from ..shell import call, scp
from ..exceptions import CommandExecutionError
from ..shell import tmpdir

# constants

# code
class Unification (Action):
    r"""
    Re-samples and co-registers images to a common space.
    """
    
    BSPLINE_ORDER = 3
    """The order of the bspline-interpolation employed during the resampling."""
    ACTIONDIR = '01unification'
    """The default action working directory name."""
    
    def __init__(self, cwd, imageset, fixedsequence = 'flair', targetspacing = 1):
        """
        Parameters
        ----------
        cwd : string
            The working directory in which to execute the action.
        imageset : ImageSet
            The input image set.
        fixedsequence : string
            The sequence which is re-sampled and serves as fixed image during
            registration. Must be contained in ``imageset``.
        targetspacing : False or number or sequence of numbers
            The target spacing for all images. If ``False``, the original spacing of the
            ``fixedsequence`` image is kept; if a single number, isotropic spacing is
            assumed; a sequence of numbers denotes custom spacing.
        """
        super(Unification, self).__init__(cwd)
        self.inset = imageset
        self.inset.validate()
        self.fixedsequence = fixedsequence
        self.targetspacing = targetspacing # False = no re-sampling, single number = isotropic, sequence of numbers = custom
        
    @property
    def cawd(self):
        """
        Returns
        -------
        The current action working directory.
        """
        return os.path.join(self.cwd, Unification.ACTIONDIR)
        
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
        for case, filename in self.inset.iterfilenames(self.fixedsequence):
            src = self.inset.getfilebyfilename(case, filename)
            trg = self.outset.getfilebyfilename(case, filename)
            if self.targetspacing: # re-sample
                self.tasks.append([[src], [trg], self.__resample, [src, trg], dict(), 're-sample'])
            else: # simply copy
                self.tasks.append([[src], [trg], self.__scp, [src, trg], dict(), 'save-copy'])   

        # Registration
        remaining_sequences = set(self.inset.sequences).difference([self.fixedsequence])
        for case, filename in self.inset.iterfilenames(remaining_sequences):
            moving = self.inset.getfilebyfilename(case, filename)
            fixed = self.outset.getfilebysequence(case, self.fixedsequence)
            to = self.outset.getfilebyfilename(case, filename)
            self.tasks.append([[moving, fixed], [to], self.__register, [fixed, moving, to], dict(), 'rigid-registration'])
                
    def _postprocess(self):
        self.outset.validate()
        
    def __resample(self, src, trg):
        img, hdr = medpy.io.load(src)
        img, hdr = resample(img, hdr, self.targetspacing, Unification.BSPLINE_ORDER)
        medpy.io.save(img, trg, hdr)   
        
    def __register(self, fixed, moving, to):
        """
        Rigidly registers the ``moving``image to the ``fixed`` image using *elastix*, saving it as ``to``.
        
        Parameters
        ----------
        fixed : string
            Path to the fixed image.
        moving : string
            Path to the moving image.
        to : string
            The file where to put the registered image.
        """
        # with temporary directory
        with tmpdir() as t:
            # prepare file paths
            cnf_file = os.path.join(t, 'rigid_cnf.txt')
            result_file = os.path.join(t, 'result.0.nii.gz')
            transformation_file = os.path.join(t, 'TransformParameters.0.txt')
            transformation_file_to = '{}.transparameters.txt'.format(to)
            
            # create configuration file
            with open(cnf_file, 'w') as f:
                f.write(self.ELASTIX_RIGID_REGISTRATION_CNF)
                
            # prepare and run registration command
            cmd = ['elastix', '-f', fixed, '-m', moving, '-out', t, '-p', cnf_file, '-threads={}'.format(self.processors)]
            rtcode, stdout, stderr = call(cmd)
            
            # check if successful
            if not os.path.isfile(result_file):
                raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Registration result image not created.')
            elif not os.path.isfile(transformation_file):
                raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Registration transformation file not created.')
            
            # copy
            scp(result_file, to)
            scp(transformation_file, transformation_file_to)
        
    ELASTIX_RIGID_REGISTRATION_CNF = \
"""
// Example parameter file for rotation registration
// C-style comments: //

// The internal pixel type, used for internal computations
// Leave to float in general. 
// NB: this is not the type of the input images! The pixel 
// type of the input images is automatically read from the 
// images themselves.
// This setting can be changed to "short" to save some memory
// in case of very large 3D images.
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")

// The dimensions of the fixed and moving image
// NB: This has to be specified by the user. The dimension of
// the images is currently NOT read from the images.
// Also note that some other settings may have to specified
// for each dimension separately.
(FixedImageDimension 3)
(MovingImageDimension 3)

// Specify whether you want to take into account the so-called
// direction cosines of the images. Recommended: true.
// In some cases, the direction cosines of the image are corrupt,
// due to image format conversions for example. In that case, you 
// may want to set this option to "false".
(UseDirectionCosines "true")

// **************** Main Components **************************

// The following components should usually be left as they are:
(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")

// These may be changed to Fixed/MovingSmoothingImagePyramid.
// See the manual.
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")

// The following components are most important:
// The optimizer AdaptiveStochasticGradientDescent (ASGD) works
// quite ok in general. The Transform and Metric are important
// and need to be chosen careful for each application. See manual.
(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "EulerTransform")
(Metric "AdvancedMattesMutualInformation")

// ***************** Transformation **************************

// Scales the rotations compared to the translations, to make
// sure they are in the same range. In general, it's best to  
// use automatic scales estimation:
(AutomaticScalesEstimation "true")

// Automatically guess an initial translation by aligning the
// geometric centers of the fixed and moving.
(AutomaticTransformInitialization "true")

// Whether transforms are combined by composition or by addition.
// In generally, Compose is the best option in most cases.
// It does not influence the results very much.
(HowToCombineTransforms "Compose")

// ******************* Similarity measure *********************

// Number of grey level bins in each resolution level,
// for the mutual information. 16 or 32 usually works fine.
// You could also employ a hierarchical strategy:
//(NumberOfHistogramBins 16 32 64)
(NumberOfHistogramBins 32)

// If you use a mask, this option is important. 
// If the mask serves as region of interest, set it to false.
// If the mask indicates which pixels are valid, then set it to true.
// If you do not use a mask, the option doesn't matter.
(ErodeMask "false")

// ******************** Multiresolution **********************

// The number of resolutions. 1 Is only enough if the expected
// deformations are small. 3 or 4 mostly works fine. For large
// images and large deformations, 5 or 6 may even be useful.
(NumberOfResolutions 4)

// The downsampling/blurring factors for the image pyramids.
// By default, the images are downsampled by a factor of 2
// compared to the next resolution.
// So, in 2D, with 4 resolutions, the following schedule is used:
//(ImagePyramidSchedule 8 8  4 4  2 2  1 1 )
// And in 3D:
//(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 )
// You can specify any schedule, for example:
//(ImagePyramidSchedule 4 4  4 3  2 1  1 1 )
// Make sure that the number of elements equals the number
// of resolutions times the image dimension.

// ******************* Optimizer ****************************

// Maximum number of iterations in each resolution level:
// 200-500 works usually fine for rigid registration.
// For more robustness, you may increase this to 1000-2000.
(MaximumNumberOfIterations 250)

// The step size of the optimizer, in mm. By default the voxel size is used.
// which usually works well. In case of unusual high-resolution images
// (eg histology) it is necessary to increase this value a bit, to the size
// of the "smallest visible structure" in the image:
//(MaximumStepLength 1.0)

// **************** Image sampling **********************

// Number of spatial samples used to compute the mutual
// information (and its derivative) in each iteration.
// With an AdaptiveStochasticGradientDescent optimizer,
// in combination with the two options below, around 2000
// samples may already suffice.
(NumberOfSpatialSamples 2048)

// Refresh these spatial samples in every iteration, and select
// them randomly. See the manual for information on other sampling
// strategies.
(NewSamplesEveryIteration "true")
(ImageSampler "Random")

// ************* Interpolation and Resampling ****************

// Order of B-Spline interpolation used during registration/optimisation.
// It may improve accuracy if you set this to 3. Never use 0.
// An order of 1 gives linear interpolation. This is in most 
// applications a good choice.
(BSplineInterpolationOrder 1)

// Order of B-Spline interpolation used for applying the final
// deformation.
// 3 gives good accuracy; recommended in most cases.
// 1 gives worse accuracy (linear interpolation)
// 0 gives worst accuracy, but is appropriate for binary images
// (masks, segmentations); equivalent to nearest neighbor interpolation.
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

// Choose whether to generate the deformed moving image.
// You can save some time by setting this to false, if you are
// only interested in the final (nonrigidly) deformed moving image
// for example.
(WriteResultImage "true")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "nii.gz")
"""
"""The elastix rigid registration configuration file."""
            

        