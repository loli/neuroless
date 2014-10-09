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
# since 2014-10-01
# status Development

# build-in module
import os
import itertools
import multiprocessing

# third-party modules
from medpy import filter
from medpy.io import header, load, save

# own modules
from .. import FileSet, TaskMachine
from ..shell import tmpdir, call, scp
from ..exceptions import CommandExecutionError

# constants (see end of file for more constants)

# code
def unify(directory, inset, fixedsequence = 'flair', targetspacing = 1):
    r"""
    Re-sample and co-register images to a common space.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    fixedsequence : string
        The sequence which is re-sampled and serves as fixed image during
        registration. Must be contained in ``fileset``.
    targetspacing : False or number or sequence of numbers
        The target spacing for all images. If ``False``, the original spacing of the
        ``fixedsequence`` image is kept; if a single number, isotropic spacing is
        assumed; a sequence of numbers denotes custom spacing.
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and representing the processed images.
    """
    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
        
    # prepare output file set
    resultset = FileSet.fromfileset(directory, inset)

    # prepare and register re-sampling tasks
    for case in inset.cases:
        src = inset.getfile(case=case, identifier=fixedsequence)
        dest = resultset.getfile(case=case, identifier=fixedsequence)
        if targetspacing: # re-sample
            tm.register([src], [dest], sresample, [src, dest, targetspacing], dict(), 're-sample')
        else: # simply copy
            tm.register([src], [dest], scp, [src, dest], dict(), 'secure-copy')
            
    # run
    tm.run()            

    # prepare and register registration task
    for case, identifier in itertools.product(inset.cases, inset.identifiers):
        if identifier == fixedsequence: continue
        moving = inset.getfile(case=case, identifier=identifier)
        fixed = resultset.getfile(case=case, identifier=fixedsequence)
        dest = resultset.getfile(case=case, identifier=identifier)
        tm.register([moving, fixed], [dest], register, [fixed, moving, dest], dict(), 'rigid-registration')
       
    # run
    tm.run()
    
    return resultset

def resamplebyexample(directory, inset, referenceset, referenceidentifier = False, binary = False):
    r"""
    Re-sample binary images to a new spacing, origin and size.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    referenceset : FileSet
        Images displaying the target resolutions ``inset``.
    referenceidentifier : FileSet
        Identifier to extract the right images from referenceset.
    binary : bool
        Set to ``True`` for binary images.
                
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and representing the processed images.
    """
    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
        
    # prepare output file set
    resultset = FileSet.fromfileset(directory, inset)

    # prepare and register re-sampling tasks
    for case in inset.cases:
        src = inset.getfile(case=case)
        referencefile = referenceset.getfile(case=case, identifier=referenceidentifier)
        dest = resultset.getfile(case=case)
        tm.register([src, referencefile], [dest], sresamplebyexample, [src, dest, referencefile, binary], dict(), 'binary re-sample')

    # run
    tm.run()

    return resultset        

def resample(directory, inset, targetspacing = 1, order = 3):
    r"""
    Re-sample images to a new spacing.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    targetspacing : False or number or sequence of numbers
        The target spacing for all images. If ``False``, the original spacing of the
        ``fixedsequence`` image is kept; if a single number, isotropic spacing is
        assumed; a sequence of numbers denotes custom spacing.
    order : integer
        The order of the b-spline-re-sampling. Set to 1 for binary images.
                        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and representing the processed images.
    """    
    # prepare the task machine
    tm = TaskMachine(multiprocessing=True)
        
    # prepare output file set
    resultset = FileSet.fromfileset(directory, inset)

    # prepare and register re-sampling tasks
    for case in inset.cases:
        src = inset.getfile(case=case)
        dest = resultset.getfile(case=case)
        if targetspacing: # re-sample
            tm.register([src], [dest], sresample, [src, dest, targetspacing, order], dict(), 're-sample')
        else: # simply copy
            tm.register([src], [dest], scp, [src, dest, order], dict(), 'secure-copy')

    # run
    tm.run()

    return resultset
        
def sresample(src, dest, spacing, order = 3):
    r"""
    Secure-re-sample an image located at ``src`` to ``spacing`` and save it under
    ``dest``.
    
    Parameters
    ----------
    src : string
        Source image file.
    dest : string
        Destination image file.
    spacing : sequence of numbers
        The target voxel spacing.
    order : integer
        The b-spline-order as used by `scipy.ndimage.zoom`.
    """
    img, hdr = load(src)
    img, hdr = filter.resample(img, hdr, spacing, order)
    save(img, dest, hdr)           
        
def sresamplebyexample(src, dest, referenceimage, binary = False):
    r"""
    Secure-re-sample an image located at ``src`` by example ``referenceimage`` and
    save it under ``dest``.
    
    Parameters
    ----------
    src : string
        Source image file.
    dest : string
        Destination image file.
    referenceimage : string
        Reference image displaying the target spacing, origin and size.
    binary : bool
        Set to ``True`` for binary images.
    """
    # get target voxel spacing
    refimage, refhdr = load(referenceimage)
    spacing = header.get_pixel_spacing(refhdr)
    
    with tmpdir() as t:
        # create a temporary copy of the reference image with the source image data-type (imiImageResample requires both images to be of the same dtype)
        srcimage, _ = load(src)
        save(refimage.astype(srcimage.dtype), os.path.join(t, 'ref.nii.gz'), refhdr)
    
        # prepare and run registration command
        cmd = ['imiImageResample', '-I', src, '-O', dest, '-R', os.path.join(t, 'ref.nii.gz'), '-s'] + map(str, spacing)
        if binary:
            cmd += ['-b']
        rtcode, stdout, stderr = call(cmd)
    
    # check if successful
    if not os.path.isfile(dest):
        raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Binary re-sampling result image not created.')
        
def register(fixed, moving, dest):
    r"""
    Rigidly registers the ``moving``image to the ``fixed`` image using *elastix*, saving
    it under ``dest`.
    
    Parameters
    ----------
    fixed : string
        Path to the fixed image.
    moving : string
        Path to the moving image.
    dest : string
        The file where to put the registered image.
    """
    # with temporary directory
    with tmpdir() as t:
        # prepare file paths
        cnf_file = os.path.join(t, 'rigid_cnf.txt')
        result_file = os.path.join(t, 'result.0.nii.gz')
        transformation_file = os.path.join(t, 'TransformParameters.0.txt')
        transformation_file_to = '{}.transparameters.txt'.format(dest)
        
        # create configuration file
        with open(cnf_file, 'w') as f:
            f.write(ELASTIX_RIGID_REGISTRATION_CNF)
            
        # prepare and run registration command
        cmd = ['elastix', '-f', fixed, '-m', moving, '-out', t, '-p', cnf_file, '-threads={}'.format(multiprocessing.cpu_count())]
        rtcode, stdout, stderr = call(cmd)
        
        # check if successful
        if not os.path.isfile(result_file):
            raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Registration result image not created.')
        elif not os.path.isfile(transformation_file):
            raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Registration transformation file not created.')
        
        # copy
        scp(result_file, dest)
        scp(transformation_file, transformation_file_to)
        
ELASTIX_RIGID_REGISTRATION_CNF = \
"""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")
(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "EulerTransform")
(Metric "AdvancedMattesMutualInformation")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "true")
(HowToCombineTransforms "Compose")
(NumberOfHistogramBins 32)
(NumberOfResolutions 4)
(MaximumNumberOfIterations 250)
(NumberOfSpatialSamples 2048)
(NewSamplesEveryIteration "true")
(ImageSampler "Random")
(BSplineInterpolationOrder 1)
(FinalBSplineInterpolationOrder 3)
(DefaultPixelValue 0)
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "nii.gz")
"""
"""The elastix rigid registration configuration file."""
            

        