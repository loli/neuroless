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
# since 2014-09-29
# status Development

# build-in module
import os
import pickle
from neuroless import ImageSet
from neuroless.exceptions import ConsistencyError

# third-party modules

# own modules

# constants

# code
class TrainedForestInstance (object):
    """
    Management object for storing and retrieving a trained decision forest and additional components.
    """
    
    FILENAME_CONFIG = 'config.pkl'
    FILENAME_FOREST = 'forest.pkl'
    FILENAME_INTSTDMODEL_BASESTRING = 'intstdmodel_{}.pkl'
    
    def __init__(self, directory, sequences):
        """
        Create a new instance, ready to be filled with all necessary values at ``dir``.
        
        Parameters
        ----------
        directory : string
            Empty and existing directory in which to base this instance.
        sequences : sequence of strings
            The MRI instances this forest was trained with.
        """
        # check directory for existence
        if not os.path.isdir(directory):
            raise ValueError('The directory "{}" does not exist.'.format(directory))
        
        self.__directory = directory
        self.__sequences = list(sequences)
        
        self.__forestfile = os.path.join(self.directory, TrainedForestInstance.FILENAME_FOREST)
        self.__configfile = os.path.join(self.directory, TrainedForestInstance.FILENAME_CONFIG)
        
    @staticmethod
    def fromdirectory(directory):
        """
        Create an instance from a trained forest instance stored in ``directory``.
        
        Parameters
        ----------
        directory : string
            Location of the stored trained forest.
            
        Returns
        -------
        trainedforest : TrainedForestInstance
            An instance of this class pointing to the supplied directory and with all variables set.
        """
        # parse the config file
        cnffile = os.path.join(directory, TrainedForestInstance.FILENAME_CONFIG)
        sequences, skullstripsequence, samplingparameters, forestparameters, \
               basesequence, workingresolution, trainingimages = TrainedForestInstance.__parse_config(cnffile)
        # create new instance
        tfi = TrainedForestInstance(directory, sequences)
        # set the configuration parameters
        tfi.skullstripsequence = skullstripsequence
        tfi.samplingparameters = samplingparameters
        tfi.forestparameters = forestparameters
        tfi.basesequence = basesequence
        tfi.workingresolution = workingresolution
        tfi.__trainingimages = trainingimages
        # validate
        tfi.validate()
        
        return tfi

    @property
    def forest(self):
        """
        The decision forest object.
        """
        with open(self.__forestfile, 'rb') as f:
            return pickle.load(f)
            
    @forest.setter
    def forest(self, forest):
        if os.path.exists(self.__forestfile):
            raise ValueError('"{}" already exists.'.format(self.__forestfile))
        with open(self.__forestfile, 'wb') as f:
            pickle.dump(forest, f)
        
    @property
    def sequences(self):
        """
        The sequences supported by the forest.
        """
        return self.__sequences
        
    @property
    def directory(self):
        """
        The directory this forest is based in.
        """
        return self.__directory
        
    @property
    def trainingimages(self):
        """
        The images used to train this trained forest instance (optional).
        """
        return self.__trainingimages
        
    @trainingimages.setter
    def trainingimages(self, i):
        # check if i is ImageSet instance
        if not isinstance(i, ImageSet):
            raise ValueError('The passed training images must be contained in an ImageSet object.')
        # check if is valid
        i.validate()   
        # then convert to internal format
        self.__trainingimages = [f for f in i.iterfiles(self.sequences)]
        
    def validate(self):
        """
        Check if all information required to persist a trained forest instance are present.
        """
        # required are
        if not hasattr(self, 'samplingparameters'):
            raise ConsistencyError('"samplingparameters" not set.')
        if not hasattr(self, 'forestparameters'):
            raise ConsistencyError('"forestparameters" not set.')
        if not hasattr(self, 'basesequence'):
            raise ConsistencyError('"basesequence" not set.')
        if not hasattr(self, 'workingresolution'):
            raise ConsistencyError('"workingresolution" not set.')
        if not hasattr(self, 'skullstripsequence'):
            raise ConsistencyError('"skullstripsequence" not set.')
        # optional instance attribute: self.trainingimages
        # files which must exist
        if not os.path.isfile(self.__forestfile):
            raise ConsistencyError('No forest set ("{}" does not exist.'.format(self.__forestfile))
        for sequence in self.sequences:
            mfname = self.__getintensitystdmodelfile(sequence) 
            if not os.path.isfile(mfname):
                raise ConsistencyError('Model file for sequence "" missing ("{}" does not exist.'.format(sequence, mfname))
        
    def persist(self):
        """
        Persist the trained forest instance.
        """
        # call validate
        self.validate()
        # save the config (if not already there)
        self.__persist_config() #!TODO: write this method!
        
    def prettyinfo(self):
        """
        Returns a pretty-formatted string containing all the characteristics of the trained forest instance.
        """
        self.validate()
        
        base = """
        Representation of a trained forest instance located under "{directory}"
        #######################################################################
        Sequences covered: {sequences}
        Base-sequence, to which all others are registered: {basesequence}
        Resolution, to which the base-sequence is re-sampled beforehand: {workingresolution}
        Sequence used to the skull-stripping: {skullstripsequence}
        
        Sampling parameters employed:
        {samplingparameters}
        
        Forest training parameters employed:
        {forestparameters}
        
        Configuration file: {configfile}
        Forest file: {forestfile}
        Intensity range standardisation model files: {modelfiles}
        
        Training images used to train this model (optional parameter):
        {trainingimages}
        """
        
        return base.format(directory = self.directory,
                           sequences = self.sequences,
                           basesequence = self.basesequence,
                           workingresolution = self.workingresolution,
                           skullstripsequence = self.skullstripsequence,
                           samplingparameters = self.samplingparameters,
                           forestparameters = self.forestparameters,
                           configfile = self.__configfile,
                           forestfile = self.forestfile,
                           modelfiles = [self.__getintensitystdmodelfile(s) for s in self.sequences],
                           trainingimages = ['\n'.join(self.trainingimages)])
        
        
    def getintensitystdmodel(self, sequence):
        """
        Get the intensity standardisation model for the ``sequence``.
        
        Parameters
        ----------
        sequence : string
            A valid MRI sequence for this trained forest instance.
            
        Returns
        -------
        model : IntensityRangeStandardisation
            The corresponding model.
        """
        # check if sequence in self.sequences
        if not sequence in self.sequences:
            raise ValueError('Sequence "{}" unknown, must be one of "{}".'.format(sequence, self.sequences))
        # un-pickel model file
        with open(self.__getintensitystdmodelfile(sequence), 'rb') as f:
            return pickle.load(f)
        
    def setintensitystdmodel(self, sequence, model):
        """
        Set the intensity standardisation model for the ``sequence``.
        
        Parameters
        ----------
        sequence : string
            A valid MRI sequence for this trained forest instance.
        model : IntensityRangeStandardisation
            The corresponding model.
        """
        # check if sequence in self.sequences
        if not sequence in self.sequences:
            raise ValueError('Sequence "{}" unknown, must be one of "{}".'.format(sequence, self.sequences))
        # check if model file already exists
        mfname = self.__getintensitystdmodelfile(sequence)
        if os.path.exists(mfname):
            raise ValueError('"{}" already exists.'.format(mfname))
        # pickle model file
        with open(mfname, 'wb') as f:
            pickle.dump(model, f)
            
    def __getintensitystdmodelfile(self, sequence):
        """
        Returns the model file associated with a sequence.
        """
        if not sequence in self.sequences:
            raise ValueError('Sequence "{}" unknown, must be one of "{}".'.format(sequence, self.sequences))
        return os.path.join(self.directory, TrainedForestInstance.FILENAME_INTSTDMODEL_BASESTRING.format(sequence))            

    def __persist_config(self):
        """
        Persist the configuration into a file.
        """
        if os.path.exists(self.__configfile):
            raise ValueError('"{}" already exists.'.format(self.__configfile))
        with open(self.__configfile, 'wb') as f:
            pickle.dump(self.sequences, f)
            pickle.dump(self.skullstripsequence, f)
            pickle.dump(self.samplingparameters, f)
            pickle.dump(self.forestparameters, f)
            pickle.dump(self.basesequence, f)
            pickle.dump(self.workingresolution, f)
            if not hasattr(self, 'trainingimages'):
                self.trainingimages = []
            pickle.dump(self.trainingimages, f)
            
    @staticmethod
    def __parse_config(cnffile):
        """
        Parse a configuration file and return the configuration.
        """
        if not os.path.exists(cnffile):
            raise ValueError('"{}" does not exist.'.format(cnffile))
        with open(cnffile, 'rb') as f:
            sequences = pickle.load(f)
            skullstripsequence = pickle.load(f)
            samplingparameters = pickle.load(f)
            forestparameters = pickle.load(f)
            basesequence = pickle.load(f)
            workingresolution = pickle.load(f)
            trainingimages = pickle.load(f)
        return sequences, skullstripsequence, samplingparameters, forestparameters, \
               basesequence, workingresolution, trainingimages