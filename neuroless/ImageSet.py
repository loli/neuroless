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
import medpy.io
import itertools
from .exceptions import ConsistencyError, AttributeSetterError
from .shell import mkdircond

# own modules

# constants

# code
class ImageSet (object):
    r"""
    Wrapper for an image set on hard drive.
    """
    
    # properties
    @property
    def directory(self):
        """The imageset's directory."""
        return self._directory
    
    def _setdirectory(self, value):
        """
        Set the `~ImageSet.directory` attribute to ``value`` and create it on hard drive.
        
        Can only be called / set once.
        
        Parameters
        ----------
        value : string
            Path pointing to an (possibly non-existent) directory.
        """
        if hasattr(self, 'directory'):
            raise AttributeSetterError('The "directory" attribute can be set only once.')
        mkdircond(value)
        self._directory = value
    
    @property
    def cases(self):
        """The imageset's cases."""
        return self._cases
    
    def _setcases(self, value):
        """
        Set the `~ImageSet.cases` attribute to ``value`` and creates them on hard drive.
        
        Can only be called / set once. Can only be called after `~ImageSet.directory` is set.
        
        Parameters
        ----------
        value : sequence of strings
            The case names.
        """
        if hasattr(self, 'cases'):
            raise AttributeSetterError('The "cases" attribute can be set only once.')
        elif not hasattr(self, 'directory'):
            raise AttributeSetterError('The "directory" has to be set before the "cases" attribute.')
        for case in value:
            mkdircond(os.path.join(self.directory, case))
        self._cases = value
    
    @property
    def sequences(self):
        """The sequences contained."""
        return self._sequences
    
    @property
    def files(self):
        """The file names contained."""
        return self._files
    
    @property
    def sequences_to_files(self):
        """The sequences to files mapping."""
        return dict(zip(self.sequences, self.files))
    
    @property
    def files_to_sequences(self):
        """The files to sequences mapping."""
        return dict(zip(self.files, self.sequences))
    
    def _setsequencesfiles(self, sequences, files):
        """
        Sets the `~ImageSet.sequences` and `~ImageSet.files` attributes and checks for consistent.
        
        sequences : sequence of strings
            The sequences.
        files : sequence of strings
            The file names. Must be of same length and in corresponding order to ``sequences``.
        """
        if not len(files) == len(sequences):
            raise ConsistencyError('The number of files ({}) does not correspond with the number of sequences ({}).'.format(len(files), len(sequences)))
        self._sequences = sequences
        self._files = files
    
    # constructors

     
    def __init__(self, directory, cases, sequences, files):
        """
        Initialize a new, empty image set in ``directory`` with the directory structure created.
        
        Parameters
        ----------
        directory : string
            The base directory of the image set.
        cases : sequence of strings
            The cases.
        sequences : sequence of strings
            The sequences.
        files : sequence of strings
            The file names. Must be of same length and in corresponding order to ``sequences``.
        """
        self._setdirectory(directory)
        self._setcases(cases)
        self._setsequencesfiles(sequences, files)
        
    @staticmethod
    def fromdirectory(directory, sequences):
        """
        Create a new image set from an existing structure in ``directory``.
        
        Parameters
        ----------
        directory : string
            A directory containing an image set with the appropriate structure.
        sequences : sequence fo strings
            The sequences to which the (alphabetically ordered) image file names should be mapped.
            
        Returns
        -------
        imageset : ImageSet
            A new imageset representing ``directory``.        
        """
        # modify sequences
        sequences = map(lambda x: x.lower(), sequences)
        # check image set main directory
        ImageSet.validate_directory(directory)
        # get cases
        _, cases, _ = os.walk(directory).next()
        cases = sorted(cases)
        # get files
        _, _, files = os.walk(os.path.join(directory, cases[0])).next()
        files = sorted(files) # os.walk return the image in arbitrary order
        # create and return ImageSet object
        return ImageSet(directory, cases, sequences, files) 
        
    @staticmethod
    def fromimageset(directory, imageset):
        """
        Create a new image set at in ``directory`` after the example of ``imageset``.
        
        Parameters
        ----------
        directory : string
            The base directory of the image set.
        imageset : ImageSet
            An `ImageSet` instance from which to copy the image set structure.
            
        Returns
        -------
        imageset : ImageSet
            A new image set with the directory structure created but void of images.
        """
        return ImageSet(directory, imageset.cases, imageset.sequences, imageset.files)
    
    # iterators
    def iterfilenames(self, sequences = False):
        """
        Get an iterator over the image file names.
        
        Parameters
        ----------
        sequences : False or string or sequence of strings.
            Supply a sequence or list of sequences to iterate over this sequences only.
            
        Returns
        -------
        case : string
            The case.
        filename : string
            The file names only.
        """
        return ImageSet.IterFileNames(self, sequences)    
    
    def iterfiles(self, sequences = False):
        """
        Get an iterator over the image files.
        
        Parameters
        ----------
        sequences : False or string or sequence of strings.
            Supply a sequence or list of sequences to iterate over this sequences only.
            
        Returns
        -------
        case : string
            The case.
        filename : string
            The complete path to the images.
        """
        return ImageSet.IterFiles(self, sequences)
    
    def iterimages(self, sequences = False):
        """
        Get an iterator over the (loaded) images.
        
        Parameters
        ----------
        sequences : False or string or sequence of strings.
            Supply a sequence or list of sequences to iterate over this sequences only.
            
        Returns
        -------
        case : string
            The case.
        img, hdr : ndarray, object
            The image-data / image-header pairs in ``case``.
        """
        return ImageSet.IterImages(self, sequences)  
    
    def itercasefiles(self, case, sequences = False):
        """
        Get an iterator over the image files in a case.
        
        Parameters
        ----------
        case : string
            The case name.
        sequences : False or string or sequence of strings.
            Supply a sequence or list of sequences to iterate over this sequences only.
            
        Returns
        -------
        filename : string
            The complete path to the images in ``case``. 
        """
        return ImageSet.IterCaseFiles(self, case, sequences)
    
    def itercaseimages(self, case, sequences = False):
        """
        Get an iterator over the (loaded) images in a case.
        
        Parameters
        ----------
        case : string
            The case name.
        sequences : False or string or sequence of strings.
            Supply a sequence or list of sequences to iterate over this sequences only.
            
        Returns
        -------
        img, hdr : ndarray, object
            The image-data / image-header pairs in ``case``.
        """
        return ImageSet.IterCaseImages(self, case, sequences) 
    
    class IterFileNames (object):
        """Iterator over the image file names."""
        
        def __init__(self, outer_instance, sequences = False):
            """
            Parameters
            ----------
            sequences : False or string or sequence of strings.
                Supply a sequence or list of sequences to iterate over this sequences only.
            """
            # set sequences correctly
            if False == sequences:
                sequences = outer_instance.sequences
            elif isinstance(sequences, basestring):
                sequences = (sequences, )
            
            # create the file list
            filenames = [outer_instance.sequences_to_files[seq] for seq in sequences]
            self._case_files = list(itertools.product(outer_instance.cases, filenames))
                    
        def __iter__(self):
            return self
 
        def next(self):
            if 0 == len(self._case_files):
                raise StopIteration
            else:
                return self._case_files.pop()
    
    class IterFiles (object):
        """Iterator over the image files."""
        
        def __init__(self, outer_instance, sequences = False):
            """
            Parameters
            ----------
            sequences : False or string or sequence of strings.
                Supply a sequence or list of sequences to iterate over this sequences only.
            """
            # set sequences correctly
            if False == sequences:
                sequences = outer_instance.sequences
            elif isinstance(sequences, basestring):
                sequences = (sequences, )
            
            # create the file list
            filenames = [outer_instance.sequences_to_files[seq] for seq in sequences]
            self._case_files = [(casename, os.path.join(outer_instance.directory, casename, filename)) for filename in filenames for casename in outer_instance.cases]
                    
        def __iter__(self):
            return self
 
        def next(self):
            if 0 == len(self._case_files):
                raise StopIteration
            else:
                return self._case_files.pop()
            
    class IterImages (IterFiles):
        """Iterator over the (loaded) images."""

        def __init__(self, outer_instance, sequences = False):
            """
            Parameters
            ----------
            sequences : False or string or sequence of strings.
                Supply a sequence or list of sequences to iterate over this sequences only.
            """
            super(ImageSet.IterImages, self).__init__(outer_instance, sequences)
            
        def __iter__(self):
            return self
        
        def next(self):
            if 0 == len(self._case_files):
                raise StopIteration
            else:
                case, filename = self._case_files.pop()
                img, hdr = medpy.io.load(filename) 
                return case, img, hdr
    
    class IterCaseFiles (object):
        """Iterator over the image files in a case."""
        
        def __init__(self, outer_instance, case, sequences = False):
            """
            Parameters
            ----------
            case : string
                The case name.
            sequences : False or string or sequence of strings.
                Supply a sequence or list of sequences to iterate over this sequences only.
            """
            # set sequences correctly
            if False == sequences:
                sequences = outer_instance.sequences
            elif isinstance(sequences, basestring):
                sequences = (sequences, )
            
            # create the file list
            filenames = [outer_instance.sequences_to_files[seq] for seq in sequences]
            self._files = [os.path.join(self.outer_instance.directory, case, filename) for filename in filenames]
                    
        def __iter__(self):
            return self
        
        def next(self):
            if 0 == len(self._files):
                raise StopIteration
            else:
                return self._files.pop()
            
    class IterCaseImages (IterCaseFiles):
        """Iterator over the (loaded) images in a case."""
        
        def __init__(self, outer_instance, case, sequences = False):
            """
            Parameters
            ----------
            case : string
                The case name.
            sequences : False or string or sequence of strings.
                Supply a sequence or list of sequences to iterate over this sequences only.
            """
            super(ImageSet.IterCaseImages, self).__init__(outer_instance, case, sequences)
            
        def __iter__(self):
            return self
        
        def next(self):
            if 0 == len(self._files):
                raise StopIteration
            else:
                return medpy.io.load(self._files.pop())
         
    def getfilebyfilename(self, case, filename):
        """
        The complete path to the file identified by ``case`` and ``filename``.
        
        Parameters
        ----------
        case : string
            A valid case.
        filename : string
            A valid filename.
            
        Returns
        -------
        file : string
            Complete path to the file.
        """
        if not case in self.cases:
            raise ConsistencyError('"{}" is not a case known to this image set'.format(case))
        if not filename in self.files:
            raise ConsistencyError('"{}" is not a filename known to this image set'.format(filename))
        return os.path.join(self.directory, case, filename)
    
    def getfilebysequence(self, case, sequence):
        """
        The complete path to the file identified by ``case`` and ``sequence``.
        
        Parameters
        ----------
        case : string
            A valid case.
        sequence : string
            A valid sequence.
            
        Returns
        -------
        file : string
            Complete path to the file.
        """
        if not case in self.cases:
            raise ConsistencyError('"{}" is not a case known to this image set'.format(case))
        if not sequence in self.sequences:
            raise ConsistencyError('"{}" is not a sequence known to this image set'.format(sequence))
        return os.path.join(self.directory, case, self.sequences_to_files[sequence])
          
    def addimagebysequence(self, case, sequence, img, hdr, force = False):
        """
        Adds and persists a new image to the image set.
        
        Parameters
        ----------
        case : string
            The case the image belongs to.
        sequence : string
            The sequence of the image.
        img : array_like
            The image data.
        hdr : object
            The image header.
        force : bool
            Silently override any existing files. 
        """
        if not case in self.cases:
            raise ConsistencyError('The case "{}" is not contained in the image set.'.format(case))
        if not sequence in self.sequences:
            raise ConsistencyError('The sequence "{}" is not contained in the image set.'.format(sequence))
        dest = os.path.join(self.directory, case, self.sequences_to_files[sequence])
        medpy.io.save(img, dest, hdr, force)
        
    def validate(self):
        """
        Validate the image set on hard drive.
        
        Raises
        ------
        StructureError
            If the directory layout differs from the expected structure.
        """
        # check image set base directory
        if not os.path.isdir(self.directory):
            raise ConsistencyError('image set directory "{}" does not exist'.format(self.directory))
        # check case directories
        _, dirnames, _ = os.walk(self.directory).next()
        if not 0 == len(set(self.cases).difference(dirnames)):
            raise ConsistencyError('the image set directory "{}" is lacking the folders for the cases "{}"'.format(self.directory, set(self.cases).difference(dirnames)))
        # check all case folders for expected files
        for dirname in self.cases:
            _, _, filenames = os.walk(os.path.join(self.directory, dirname)).next()
            if not 0 == len(set(self.files).difference(filenames)):
                raise ConsistencyError('non-consistent image files in case folder "{}": expected {}, missing {}'.format(os.path.join(self.directory, dirname), self.files, set(self.files).difference(filenames)))
        
        
    @staticmethod
    def validate_directory(directory):
        """
        Validates the supplied directory and its structure for being a valid image set candidate.
        
        Raises
        ------
        StructureError
            If the directory layout differs from the expected structure.
        """
        # check for empty directory
        if not os.path.isdir(directory):
            raise ConsistencyError('directory "{}" does not exist'.format(directory))
        # read in directories on next level
        _, dirnames, _ = os.walk(directory).next()
        # check for no sub-directories
        if 0 == len(dirnames):
            raise ConsistencyError('no case directories found under "{}"'.format(directory))
        # read expected image files from first sub-directory
        _, _, expected_filenames = os.walk(os.path.join(directory, dirnames[0])).next()
        expected_filenames = sorted(expected_filenames)
        # check all other sub-directories for the same files
        for dirname in dirnames[1:]:
            _, _, filenames = os.walk(os.path.join(directory, dirname)).next()
            filenames = sorted(filenames)
            if not filenames == expected_filenames:
                raise ConsistencyError('non-consistent image files in case folder "{}": expected {}, got {}'.format(os.path.join(directory, dirname), expected_filenames, filenames))
        
        
    
    