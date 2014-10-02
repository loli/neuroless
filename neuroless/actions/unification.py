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
# since 2014-10-01
# status Development

# build-in module
import os
import itertools
import multiprocessing

# third-party modules
from medpy import filter
import medpy.io

# own modules
from neuroless import FileSet, TaskMachine
from neuroless.shell import tmpdir, call, scp
from neuroless.exceptions import CommandExecutionError

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
    tm = TaskMachine()
        
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

def resample(directory, inset, targetspacing = 1):
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
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and representing the processed images.
    """    
    # prepare the task machine
    tm = TaskMachine()
        
    # prepare output file set
    resultset = FileSet.fromfileset(directory, inset)

    # prepare and register re-sampling tasks
    for case in inset.cases:
        src = inset.getfile(case=case)
        dest = resultset.getfile(case=case)
        if targetspacing: # re-sample
            tm.register([src], [dest], sresample, [src, dest, targetspacing], dict(), 're-sample')
        else: # simply copy
            tm.register([src], [dest], scp, [src, dest], dict(), 'secure-copy')

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
    img, hdr = medpy.io.load(src)
    img, hdr = filter.resample(img, hdr, spacing, order)
    medpy.io.save(img, dest, hdr)   
        
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
            

        