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
from medpy.filter import IntensityRangeStandardization
from medpy.io import load, save
import numpy

# own modules
from neuroless import TaskMachine, FileSet


# constants

# code
def percentilemodelapplication(directory, inset, brainmasks, models):
    r"""
    Apply intensity standardisation models for each sequence.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    brainmasks : FileSet
        The associated brain masks file set.
    models : FileSet
        The IntensityRangeStandardization model files for each sequence. 
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and containing the intensity standarised
        images.
    """
    # prepare the task machine
    tm = TaskMachine()
    
    # prepare output
    resultset = FileSet.fromfileset(directory, inset)

    # register model training & model application tasks
    for sequence in inset.identifiers:
        modelfile = models.getfile(identifier=sequence)
        for case in inset.cases:
            imagefile = inset.getfile(identifier=sequence, case=case)
            brainmaskfile = brainmasks.getfile(case=case)
            destfile = resultset.getfile(identifier=sequence, case=case)
            tm.register([imagefile, brainmaskfile, modelfile],
                        [destfile],
                        percentileintensityapplication,
                        [imagefile, brainmaskfile, destfile, modelfile],
                        dict(),
                        'intensity-standardisation')

    # run
    tm.run()
    
    return resultset    

def percentilemodelstandardisation(directory, inset, brainmasks):
    r"""
    Train and apply intensity standardisation models for each sequence.
    
    Parameters
    ----------
    directory : string
        Where to place the results.
    inset : FileSet
        The input file set.
    brainmasks : FileSet
        The associated brain masks file set.
        
    Returns
    -------
    resultset : FileSet
        A FileSet centered on ``directory`` and containing the intensity standarised
        images.
    models : FileSet
        A FileSet centered on ``directory`` and containing a trained model for each
        MR sequence contained in ``inset``.
    """
    # prepare the task machine
    tm = TaskMachine()
    
    # prepare output
    resultset = FileSet.fromfileset(directory, inset)
    models = FileSet(directory, False, inset.identifiers, ['{}.pkl'.format(_id) for _id in inset.identifiers], 'identifiers', False)
       
        
    # register model training & model application tasks
    for sequence in inset.identifiers:
        trainingfiles = inset.getfiles(identifier=sequence)
        brainmaskfiles = brainmasks.getfiles()
        destfiles = resultset.getfiles(identifier=sequence)
        destmodel = models.getfile(identifier=sequence)
        tm.register(trainingfiles + brainmaskfiles,
                    [destmodel] + destfiles,
                    percentileintensitystd,
                    [trainingfiles, brainmaskfiles, destfiles, destmodel],
                    dict(),
                    'intensity-standardisation')

    # run
    tm.run()
    
    return resultset, models
        
def percentileintensityapplication(imagefile, brainmaskfile, destfile, modelfile):
    """
    Apply an intensity range standardisation model to an image.
    
    Parameters
    ----------
    imagefile : string
        Image to apply the model to.
    brainmaskfile : string
        The brain mask corresponding to ``imagefile``.
    destfile : string
        The intensity standarised target location corresponding to ``imagefile``.
    modelfile : string
        The location of the model to apply.
    """
    # loading image
    image, header = load(imagefile)
        
    # loading brainmask
    mask = load(brainmaskfile)[0].astype(numpy.bool)
    
    # load model
    with open(modelfile, 'rb') as f:
        model = pickle.load(f)
        
    # apply model
    transformed_image = model.transform(image[mask])
    
    # modify original image
    image[~mask] = 0
    image[mask] = transformed_image
    
    # save to destination
    save(image, destfile, header)
    
        
def percentileintensitystd(trainingfiles, brainmaskfiles, destfiles, destmodel):
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
    transformed_images = [condense(i) for i in transformed_images]
    
    # saving the model
    with open(destmodel, 'wb') as f:
        pickle.dump(trained_model, f)
    
    # save the transformed images
    for ti, i, m, h, dest in zip(transformed_images, images, masks, headers, destfiles):
        i[~m] = 0
        i[m] = ti
        save(i, dest, h)

def condense(img):
    """
    Apply a percentile threshold to an image, condensing all outliers to the
    percentile values 1 and 99.9 respectively.
    
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
