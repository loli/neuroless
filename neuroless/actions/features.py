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
# since 2014-10-02
# status Development

# build-in module
import pickle

# third-party modules
import numpy
from medpy.io import load, header, save
from medpy.core import Logger
from medpy.features.utilities import join, append
from medpy.features.intensity import intensities, local_mean_gauss,\
    local_histogram, centerdistance_xdminus1

# own modules
from neuroless import TaskMachine, FileSet
from neuroless.exceptions import InvalidConfigurationError

# constants (see end of file for more constants)
FEATURE_DTYPE = numpy.float32

# code
def extractfeatures(directory, inset, brainmasks, groundtruth):
    r"""
    Extract features from the images.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    brainmasks : FileSet
        The associated brain masks file set.
    groundtruthset : FileSet
        The associated ground-truth file set.
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and containing the extracted feature files.
    classes : FileSet
        A FileSet centered on ``directory`` and containing the class memberships.
    fnames : FileSet
        A FileSet centered on ``directory`` and containing the feature names.
    """
    # prepare the task machine
    tm = TaskMachine()
    
    # prepare output
    resultset = FileSet(directory, inset.cases, inset.identifiers, ['{}.npy'.format(fn) for fn in inset.filenames.values()], 'identifiers', True)
    classes = FileSet(directory, inset.cases, False, ['{}.classmembership.npy'.format(cid) for cid in inset.cases], 'cases', False)
    fnames = FileSet(directory, inset.cases, False, ['{}.featurenames.pkl'.format(cid) for cid in inset.cases], 'cases', False)
        
    # register feature extraction tasks
    for case in inset.cases:
        brainmaskfile = brainmasks.getfile(case=case)
        groundtruthfile = groundtruth.getfile(case=case)
        cmdestfile = classes.getfile(case=case)
        fndestfile = fnames.getfile(case=case)
            
        tm.register(inset.getfiles(case=case) + [brainmaskfile, groundtruthfile],
                    resultset.getfiles(case=case) +[cmdestfile, fndestfile],
                    extract,
                    [inset.getfiles(case=case), resultset.getfiles(case=case), brainmaskfile, groundtruthfile, cmdestfile, fndestfile],
                    dict(),
                    'feature-extraction')
        
    # run
    tm.run()
        
    return resultset, classes, fnames
        
def extract(imagefiles, destfiles, brainmaskfile, groundtruthfile, cmdestfile, fndestfile):
    """
    Extract all features from the supplied image.
    
    Parameters
    ----------
    imagefiles : sequence of strings
        The images from which to extract the features.
    destfiles : sequence of strings
        The file in which to save the extracted features per images.
    brainmaskfile : string
        The corresponding brain mask.
    groundtruthfile : string
        The corresponding ground-truth file.
    cmdestfile : string
        The destination file for the class memberships.
    fndestfile : string
        The deatination file for the feature names.
    """
    # loading the support images
    msk = load(brainmaskfile)[0].astype(numpy.bool)
    gt = load(groundtruthfile)[0].astype(numpy.bool)
    
    # for each pair of image and destination files
    for imagefile, destfile in zip(imagefiles, destfiles):
        
        # prepare feature vector and the feature identification list
        feature_vector = None
        feature_names = []        
        
        # load the image
        img, hdr = load(imagefile)
        
        # iterate the features to extract
        for function_call, function_arguments, voxelspacing in FEATURE_CONFIG:
            
            # extract the feature
            call_arguments = list(function_arguments)
            if voxelspacing: call_arguments.append(header.get_pixel_spacing(hdr))
            call_arguments.append(msk)
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
            numpy.save(f, feature_vector.astype(FEATURE_DTYPE))
    
    # save the feature names (only once, at the end)
    with open(fndestfile, 'wb') as f:
        pickle.dump(feature_names, f)
        
    # save the class memberships (truncated by the brain mask)
    with open(cmdestfile, 'wb') as f:
        pickle.dump(gt[msk], f)

def sample(directory, features, classes, brainmasks, sampler = stratifiedrandomsampling, **kwargs):
    r"""
    Sample training set from the features.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    features : FileSet
        The feature file set.
    classes : FileSet
        The associated classes membership file set.
    brainmasks : FileSet
        The associate brain mask file set.
    samples : function
        The sample to employ.
    **kwargs
        Optional keyword arguments to be passed to the sampler.
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and containing the training set and the class membership file.
    samplepointset : FileSet
        A FileSet centered on ``directory`` and containing images denoting the samples drawn from each case.
    """
    # prepare the task machine
    tm = TaskMachine()
    
    # prepare output
    resultset = FileSet(directory, False, ['features', 'classes'], ['features.npy', 'classes.npy'], 'identifiers', False)
    samplepointset = FileSet(directory, features.cases, False, ['{}_samplepoints.nii.gz'.format(cid) for cid in features.cases], 'cases', False)
                
    # register feature sampling task
    featureclasstriples = []
    for case in features.cases:
        featurefiles = features.getfiles(case=case)
        classfile = classes.getfile(case=case)
        samplepointfile = samplepointset.getfile(case=case)
        brainmaskfile = brainmasks.getfile(case=case)
        featureclasstriples.append((featurefiles, classfile, samplepointfile))
    trainingsetfile = resultset.getfile(identifier='features')
    classfile = resultset.getfile(identifier='classes')
    tm.register(features.getfiles() + classes.getfiles() + [brainmaskfile],
                [trainingsetfile, classfile] + samplepointfile.getfiles(),
                stratifiedrandomsampling,
                [featureclasstriples, trainingsetfile, classfile, brainmaskfile], kwargs, 'sample-trainingset')

        
def stratifiedrandomsampling(featureclasstriples, trainingsetfile, classfile, brainmaskfile, nsamples = 500000, min_no_of_samples_per_class_and_case = 20):
    """
    Extract a training sample set from the supplied feature sets using stratified random sampling.
    
    Parameters
    ----------
    featureclasstriples : list of tuples
        Triples containing (a) a list of a cases feature files, (b) the corresponding
        class membership file and (c) the sample point file.
    trainingsetfile : string
        The target training set file.
    classfile : string
        The target class membership file.
    brainmaskfile : string
        The brain mask file.
    nsamples : int or False, optional
        The amount of samples to draw. If False, all are drawn.
    
    Raises
    ------
    InvalidConfigurationError
        When the current configuration would require to draw more samples than present in a case or even none.
    """
    logger = Logger.getInstance()
    
    # determine amount of samples to draw from each case
    ncases = len(featureclasstriples)
    nsamplescase = int(nsamples / ncases)
    nsampleslastcase = nsamplescase + nsamples % ncases # + a little
    logger.debug('drawing {} samples from {} cases each (total {} samples) (last case {} samples)'.format(nsamplescase, ncases, nsamples))
    
    # initialize collectors
    fg_samples = []
    bg_samples = []
    
    for cid, (featurefiles, classfile, featurepointfile) in enumerate(featureclasstriples):
        
        # adapt samples to draw from last case to draw a total of nsamples
        if len(featureclasstriples) == cid:
            nsamplescase += nsamples % ncases
        
        # load the class memberships
        classes = numpy.load(classfile, mmap_mode='r') 
        
        # determine number of fg and bg samples to draw for this case
        nbgsamples = int(classes.size / float(numpy.count_nonzero(~classes)) * nsamplescase)
        nfgsamples = int(classes.size / float(numpy.count_nonzero(classes)) * nsamplescase)
        nfgsamples += nsamplescase - (nfgsamples + nbgsamples) # +/- a little
        logger.debug('iteration {}: drawing {} fg and {} bg samples'.format(cid, nfgsamples, nbgsamples))
        
        # check for exceptions
        if nfgsamples < min_no_of_samples_per_class_and_case: raise InvalidConfigurationError('Current setting would lead to a drawing of only {} fg samples for case {}!'.format(nfgsamples, classfile))
        if nbgsamples < min_no_of_samples_per_class_and_case: raise InvalidConfigurationError('Current setting would lead to a drawing of only {} bg samples for case {}!'.format(nbgsamples, classfile))
        if nfgsamples > numpy.count_nonzero(classes):
            raise InvalidConfigurationError('Current settings would require to draw {} fg samples, but only {} present for case {}!'.format(nfgsamples, numpy.count_nonzero(classes), classfile))
        if nbgsamples > numpy.count_nonzero(~classes):
            raise InvalidConfigurationError('Current settings would require to draw {} bg samples, but only {} present for case {}!'.format(nbgsamples, numpy.count_nonzero(~classes), classfile))
        
        # get sample indices and split into fg and bg indices
        samples_indices = numpy.arange(len(classes))
        fg_samples_indices = samples_indices[classes]
        bg_samples_indices = samples_indices[~classes]
        
        # randomly draw the required number of sample indices
        numpy.random.shuffle(fg_samples_indices)
        numpy.random.shuffle(bg_samples_indices)
        fg_sample_selection = fg_samples_indices[:nfgsamples]
        bg_sample_selection = bg_samples_indices[:nbgsamples]
        
        # memory-efficient loading of the features for this case
        features = join(*[numpy.load(featurefile, mmap_mode='r') for featurefile in featurefiles])
        
        # draw and add to collection
        fg_samples.append(features[fg_sample_selection])
        bg_samples.append(features[bg_sample_selection])
        
        # create and save sample point file
        mask, maskh = load(brainmaskfile)
        mask = mask.astype(numpy.bool)
        featurepointimage = numpy.zeros_like(mask, numpy.bool)
        featurepointimage[mask][fg_sample_selection] = 1
        featurepointimage[mask][bg_sample_selection] = 2
        save(featurepointimage, featurepointfile, maskh)

    # join and append feature vectors of all cases
    fg_samples = append(*fg_samples)
    bg_samples = append(*bg_samples)
    
    # build class membership    
    samples_class_memberships = numpy.zeros(len(fg_samples) + len(bg_samples), dtype=numpy.bool)
    samples_class_memberships[:len(fg_samples)] += numpy.ones(len(fg_samples), dtype=numpy.bool)
    
    # join fg and bg feature vectors
    samples_feature_vector = append(fg_samples, bg_samples)
    
    # save all
    with open(trainingsetfile, 'wb') as f:
        numpy.save(f, samples_feature_vector)
    with open(classfile, 'wb') as f:
        numpy.save(f, samples_class_memberships)


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
