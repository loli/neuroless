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
from medpy.utilities.argparseu import existingDirectory, sequenceOfStrings

# own modules
from neuroless import FileSet
from neuroless.actions import unify, resample, stripskull, correctbiasfields, percentilemodelstandardisation, extractfeatures, startifiedrandomsampling

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
    
    # !TODO: create a target forest name and check if one of the potential output files already exists; if yes, raise Exception
    
    # check and create an image set from the training database
    traindb = FileSet.fromdirectory(args.traindb, args.sequences, filesource='identifiers')
    
    # check and create an image set from the ground-truth database
    gtset = FileSet.fromdirectory(args.groundtruthdir, traindb.cases, filesource='cases') 
    
    # check if flair sequence present
    if not BASESEQUENCE in traindb.identifiers:
        logger.error('At least one "{}" sequence must be supplied per case.'.format(BASESEQUENCE.upper()))
        sys.exit(1)
        
    # pipeline
    unified = unify(os.path.join(args.workingdir, '00unification'), traindb, fixedsequence = 'flair', targetspacing = 1)
    gtunified = resample(os.path.join(args.workingdir, '01gtunification'), gtset, targetspacing = 1)
    brainmasks = stripskull(os.path.join(args.workingdir, '02skullstrip'), unified)
    biascorrected = correctbiasfields(os.path.join(args.workingdir, '03biasfield'), unified, brainmasks)
    standarised, intstdmodels = percentilemodelstandardisation(os.path.join(args.workingdir, '04intensitystd'), biascorrected, brainmasks)
    features, classes, fnames = extractfeatures(os.path.join(args.workingdir, '05features'), standarised, brainmasks, gtunified)
    #trainingset, samplepointset = sample(os.path.join(args.workingdir, '06samplingset'), features, classes, brainmasks)
    
    # check
    for case in features.cases:
        for identifier in features.identifiers:
            print features.getfile(case=case, identifier=identifier)
    for case in classes.cases:
        print classes.getfile(case=case)
    for case in fnames.cases:
        print fnames.getfile(case=case)
        
    #print trainingset.getfiles()
    #for case in samplepointset.cases:
    #    samplepointset.getfile(case=case) 

    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    args.sequences = [s.lower() for s in args.sequences]
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('traindb', type=existingDirectory, help='The folder holding the training cases.')
    parser.add_argument('groundtruthdir', type=existingDirectory, help='The directory containing the ground-truth masks, named after the case folders.')
    parser.add_argument('targetdir', type=existingDirectory, help='The target directory in which to place the trained forest.')
    parser.add_argument('workingdir', type=existingDirectory, help='The working directory in which to place / from which to read the intermediate results.')
    parser.add_argument('sequences', type=sequenceOfStrings, help='Colon-separated list of MRI sequence names identifying the images in the traindb.')
    
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser
    
if __name__ == "__main__":
    main()
