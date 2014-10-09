#!/usr/bin/python

"""
Segment a number of formerly unseen cases.

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
import os

# third-party modules
from medpy.core import Logger
from medpy.utilities.argparseu import existingDirectory, sequenceOfStrings

# own modules
from neuroless import FileSet, TrainedForest
from neuroless.actions import unify, stripskull, correctbiasfields, percentilemodelapplication, extractfeatures, applyforest, resamplebyexample, postprocess

# information
__author__ = "Oskar Maier"
__version__ = "d0.1.0, 2014-10-06"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"
__description__ = """
                  Segment a number of formerly unseen cases.
                  
                  The cases folder has to contain one folder/case and in each folder
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
    
    # load cases
    casedb = FileSet.fromdirectory(args.cases, args.sequences, filesource='identifiers')
    
    print 'Deducted sequence to file mapping:', casedb.filenamemapping
    
    # select suitable forests
    _, forestdirs, _ = os.walk(args.forestbasedir).next()
    suitable_forests = []
    for forestdir in forestdirs:
        forest = TrainedForest.fromdirectory(os.path.join(args.forestbasedir, forestdir))
        if not set(forest.sequences).difference(args.sequences):
            suitable_forests.append(forest)
            
    # sort suitable forests by number of sequences
    suitable_forests = sorted(suitable_forests, key=lambda x: len(x.sequences))
        
    # extract configuration from most suitable forest
    forestinstance = suitable_forests[0]
    
    # pipeline: apply pre-processing steps to the cases
    print '00: Unifying MRI sequences...'
    unified = unify(os.path.join(args.workingdir, '00unification'), casedb, fixedsequence=forestinstance.fixedsequence, targetspacing=forestinstance.workingresolution)
    print '01: Computing brain masks...'
    brainmasks, _ = stripskull(os.path.join(args.workingdir, '01skullstrip'), unified, stripsequence=forestinstance.skullstripsequence)
    print '02: Computing and applying bias-field correction...'
    biascorrected = correctbiasfields(os.path.join(args.workingdir, '02biasfield'), unified, brainmasks)
    print '03: Applying intensity range models...'
    standarised = percentilemodelapplication(os.path.join(args.workingdir, '03intensitystd'), biascorrected, brainmasks, forestinstance.getintensitymodels())
    print '04: Extracting features...'
    features, _, fnames = extractfeatures(os.path.join(args.workingdir, '04features'), standarised, brainmasks)
    print '05: Segmenting cases...'
    segmentations, probabilities = applyforest(os.path.join(args.workingdir, '05segmentations'), forestinstance.forest, features, brainmasks)
    print '06: Post-processing segmentations...'
    postprocessed = postprocess(os.path.join(args.workingdir, '06postprocessed'), segmentations, args.objectthreshold)
    print '07: Re-sampling segmentations, probability maps and brain masks to original space...'
    origsegmentations = resamplebyexample(args.targetdir, postprocessed, casedb, forestinstance.fixedsequence, binary=True)
    origprobabilities = resamplebyexample(args.targetdir, probabilities, casedb, forestinstance.fixedsequence)
    origbrainmasks = resamplebyexample(os.path.join(args.targetdir, 'brainmasks'), brainmasks, casedb, forestinstance.fixedsequence, binary=True)

    print 'Successfully terminated.'

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    args.sequences = [s.lower() for s in args.sequences]
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('cases', type=existingDirectory, help='The folder holding the cases.')
    parser.add_argument('forestbasedir', type=existingDirectory, help='The folder containing the available forests.')
    parser.add_argument('targetdir', type=existingDirectory, help='The target directory in which to place the segmented images.')
    parser.add_argument('workingdir', type=existingDirectory, help='The working directory in which to place / from which to read the intermediate results.')
    parser.add_argument('sequences', type=sequenceOfStrings, help='Colon-separated list of MRI sequence names identifying the images in the traindb.')
    
    parser.add_argument('--objectthreshold', type=float, default=1500., help='Remove all binary objects of a size lower than this value in ml from the final segmentation.')
    
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser
    
if __name__ == "__main__":
    main()
