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

# third-party modules

# path changes

# own modules
from medpy.core import Logger
from medpy.utilities.argparseu import existingDirectory, sequenceOfStrings
from neuroless.ImageSet import ImageSet
import sys
from neuroless.actions import Unification, Skullstrip, Biasfield,\
    IntensityStandardisation, FeatureExtraction, Resample

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
    traindb = ImageSet.fromdirectory(args.traindb, args.sequences)
    
    # check and create an image set from the ground-truth database
    gtdb = ImageSet.fromdirectory(args.groundtruthdir, ['gt'])
    
    # check if flair sequence present
    if not BASESEQUENCE in traindb.sequences:
        logger.error('At least one "{}" sequence must be supplied per case.'.format(BASESEQUENCE.upper()))
        sys.exit(1)
    # ACTION 00: Re-sample ground-truth
    action00 = Resample(args.workingdir, gtdb, 3)
    action00.run()
    
    # ACTION 01: Unification
    action01 = Unification(args.workingdir, traindb, BASESEQUENCE, 3)
    action01.run()
    
    # ACTION 02: Skull-strip
    action02 = Skullstrip(args.workingdir, action01.getimageset())
    action02.run()
    
    # ACTION 03: Bias-field correction
    action03 = Biasfield(args.workingdir, action01.getimageset(), action02.getimageset())
    action03.run()
    
    # ACTION 04: Intensity range standardisation
    action04 = IntensityStandardisation(args.workingdir, action03.getimageset(), action02.getimageset())
    action04.run()
    
    # ACTION 05: Feature extraction
    action05 = FeatureExtraction(args.workingdir, action04.getimageset(), action02.getimageset(), action00.getimageset())
    action05.run()
    
    # check
    outset = action05.getimageset()
    for _, _file in outset.iterfiles():
        print _file

    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

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
