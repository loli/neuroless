#!/usr/bin/python

"""
Trains a decision forest on a training database.

Copyright (C) 2013 Oskar Maier

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# build-in modules
import argparse
import logging
import sys
import os

# third-party modules
from medpy.core import Logger
from medpy.utilities.argparseu import existingDirectory, sequenceOfStrings,\
    sequenceOfFloatsGt

# own modules
from neuroless import FileSet, TrainedForest
from neuroless.actions import unify, resample, stripskull, correctbiasfields, percentilemodelstandardisation, extractfeatures, sample, trainet
from neuroless.actions.features import SAMPLERS
import pickle

# information
__author__ = "Oskar Maier"
__version__ = "d0.1.0, 2014-09-22"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"
__description__ = """
                  Trains a decision forest on a training database.
                  
                  The training database has to contain one folder/case and in each folder
                  equally named images representing the MR sequences.
                                  
                  Copyright (C) 2013 Oskar Maier
                  This program comes with ABSOLUTELY NO WARRANTY; This is free software,
                  and you are welcome to redistribute it under certain conditions; see
                  the LICENSE file or <http://www.gnu.org/licenses/> for details.   
                  """

# constants
BASESEQUENCE = 'flair'

# code
def main():
    # parse cmd arguments
    parser = getParser()
    parser.parse_args()
    args = getArguments(parser)
    
    # prepare logger
    logger = Logger.getInstance()
    if args.debug: logger.setLevel(logging.DEBUG)
    elif args.verbose: logger.setLevel(logging.INFO)
    
    # create forest instance object
    forestinstance = TrainedForest(args.targetdir, args.sequences)
    
    # check and create an image set from the training database
    traindb = FileSet.fromdirectory(args.traindb, args.sequences, filesource='identifiers')
    
    # check and create an image set from the ground-truth database
    gtset = FileSet.fromdirectory(args.groundtruthdir, traindb.cases, filesource='cases') 
        
    # pipeline
    unified = unify(os.path.join(args.workingdir, '00unification'), traindb, fixedsequence=args.fixedsequence, targetspacing=args.workingresolution)
    gtunified = resample(os.path.join(args.workingdir, '01gtunification'), gtset, targetspacing=args.workingresolution)
    brainmasks = stripskull(os.path.join(args.workingdir, '02skullstrip'), unified, stripsequence=args.stripsequence)
    biascorrected = correctbiasfields(os.path.join(args.workingdir, '03biasfield'), unified, brainmasks)
    standarised, intstdmodels = percentilemodelstandardisation(os.path.join(args.workingdir, '04intensitystd'), biascorrected, brainmasks)
    features, classes, fnames = extractfeatures(os.path.join(args.workingdir, '05features'), standarised, brainmasks, gtunified)
    trainingset, samplepointset = sample(os.path.join(args.workingdir, '06samplingset'), features, classes, brainmasks, sampler=args.samplingmethod, nsamples=args.nsamples, min_no_of_samples_per_class_and_case=args.minsamplesperclassandcase)
    forest = trainet(trainingset,
                     n_estimators = args.nestimators,
                     criterion = args.criterion,
                     max_features = args.maxfeatures,
                     min_samples_split = args.minsamplesplit,
                     min_samples_leaf = args.minsamplesleaf,
                     max_depth = args.maxdepth,
                     bootstrap = args.bootstrap,
                     oob_score = args.oobscore)
    
    # set forest instance
    forestinstance.forest = forest
    forestinstance.trainingimages = traindb
    #!TODO: Get rid of these lists using sub-namespaces
    forestinstance.samplingparameters = [args.samplingmethod, args.nsamples, args.minsamplesperclassandcase]
    forestinstance.forestparameters = {'n_estimators': args.nestimators,
                                       'criterion': args.criterion,
                                       'max_features': args.maxfeatures,
                                       'min_samples_split': args.minsamplesplit,
                                       'min_samples_leaf': args.minsamplesleaf,
                                       'max_depth': args.maxdepth,
                                       'bootstrap': args.bootstrap,
                                       'oob_score': args.oobscore}
    forestinstance.fixedsequence = args.fixedsequence
    forestinstance.workingresolution = args.workingresolution
    forestinstance.skullstripsequence = args.stripsequence
    for sequence in intstdmodels.identifiers:
        model = intstdmodels.getfile(identifier=sequence)
        with open(model, 'rb') as f:
            forestinstance.setintensitystdmodel(sequence, pickle.load(f))

    print forestinstance.prettyinfo()

    # persist forest instance
    forestinstance.persist()

    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    args.sequences = [s.lower() for s in args.sequences]
    if len(args.workingresolution) == 1:
        args.workingresolution = args.workingresolution[0]
    if args.stripsequence:
        args.stripsequence = args.stripsequence.lower()
    if not args.fixedsequence in args.sequences:
        parser.error('"fixedsequence" must denote a sequence contained in the "sequences" argument.')
    if args.stripsequence and not args.stripsequence in args.sequences:
        parser.error('"stripsequence" must denote a sequence contained in the "sequences" argument.')
    if not args.maxfeatures is None:
        if 'None' == args.maxfeatures:
            args.maxfeatures = None
        elif 'auto' == args.maxfeatures:
            args.maxfeatures = 'auto'
        else:
            try:
                args.maxfeatures = int(args.maxfeatures)
            except ValueError:
                try:
                    args.maxfeatures = float(args.maxfeatures)
                except ValueError:
                    parser.error('"maxfeatures" has an invalid value')
    
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('traindb', type=existingDirectory, help='The folder holding the training cases.')
    parser.add_argument('groundtruthdir', type=existingDirectory, help='The directory containing the ground-truth masks, named after the case folders.')
    parser.add_argument('targetdir', type=existingDirectory, help='The target directory in which to place the trained forest.')
    parser.add_argument('workingdir', type=existingDirectory, help='The working directory in which to place / from which to read the intermediate results.')
    parser.add_argument('sequences', type=sequenceOfStrings, help='Colon-separated list of MRI sequence names identifying the images in the traindb.')
    
    unification = parser.add_argument_group('unification', 'The unification step re-samples and registers all MRI sequences to a common space and sequence.')
    unification.add_argument('--workingresolution', default=1, type=sequenceOfFloatsGt, help='The spacing to which all sequences are re-sampled. Can be a single number (isotropic spacing) or a colon-separated sequence of numbers. (default: 1)')
    unification.add_argument('--fixedsequence', default='flair', help='The MRI sequence to which to register the other sequences rigidly. Must be one of the sequences passed to the "sequences" argument. (default: flair)')
    
    skullstripping = parser.add_argument_group('skullstripping', 'The skull-stripping step computes a brain-mask based on the most suitable sequence.')
    skullstripping.add_argument('--stripsequence', default=False, help='The MRI sequence on which to calculate the brain mask. If in doubt, leave it to the method to choose the best sequence available. Must be one of the sequences passed to the "sequences" argument.')
    
    sampling = parser.add_argument_group('sampling', 'During the feature sampling step, a training set is sub-sampled from all available features.')
    sampling.add_argument('--samplingmethod', choices=SAMPLERS.keys(), default=SAMPLERS.keys()[0], help='The sampling method. (default: {}'.format(SAMPLERS.keys()[0]))
    sampling.add_argument('--nsamples', default=False, type=int, help='The number of sample to draw. If not supplied, all available samples are used.')
    sampling.add_argument('--minsamplesperclassandcase', default=20, type=int, help='The minimum number of samples to draw from each case and class. An exception is thrown if this is undercut. (default: 20)')
    
    training = parser.add_argument_group('training', 'The training of a decision forest can be influences with a range of parameters. if in doubt, see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html for details.')
    training.add_argument('--nestimators', default=200, type=int, help='The number of trees in the forest. (default: 200)')
    training.add_argument('--criterion', choices=['gini', 'entropy'], default='gini', help='The function to measure the quality of a split. (default: gini)')
    training.add_argument('--maxfeatures', default=None, help='The number of features to consider when looking for the best split. Can be None, auto, int or float. (default: None)')
    training.add_argument('--minsamplesplit', default=2, type=int, help='The minimum number of samples required to split an internal node. (default: 2)')
    training.add_argument('--minsamplesleaf', default=1, type=int, help='The minimum number of samples in newly created leaves. (default: 1)')
    training.add_argument('--maxdepth', default=None, type=int, help='The maximum depth of each tree. If not supplied, then nodes are expanded until another stop criteria is reached.')
    training.add_argument('--bootstrap', default=True, type=bool, help='Whether bootstrap samples are used when building trees. (default: True)')
    training.add_argument('--oobscore', default=False, type=bool, help='Whether to use out-of-bag samples to estimate the generalization error. (default: False)')
    
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser
    
if __name__ == "__main__":
    main()
