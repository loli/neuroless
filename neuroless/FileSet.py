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

# third-party modules

# own modules
from .exceptions import UnsupportedCombinationError, ConsistencyError
from .shell import mkdircond

# constants

# code
class FileSet (object):
    r"""
    Wrapper for an (ordered) set of files.
    """
    
    def __init__(self, directory, cases, identifiers, filenames, filesource = 'identifiers', subdirectories = True):
        r"""
        Parameters
        ----------
        directory : string
            Path to the directory in which to locate the file set. Created, if not
            existent.
        cases : False or sequence of strings
            The cases represented by the file set.
        identifiers : False sequence of strings
            The types represented by the file set. Usually the MRI sequences. One of
            ``cases`` and ``identifiers`` must be different from `False`.
        filesource : {'cases', 'identifiers'}
            On which to base the files. Note that that this can only be 'cases', if
            ``subdirectories`` is set to  'False'.            
        subdirectories : bool
            Whether or not to use a sub-directory structure (if yes, the cases name the
            sub-folders). Created, if not existent.
            
        Examples
        --------
        >>> fs = FileSet('/path/to/gtset/', ['01', '02'], False, ['01.nii.gz', '02.nii.gz'], False, 'cases')
        
        /path/to/gtset/01.nii.gz
        /path/to/gtset/02.nii.gz
        
        >>> fs.getfile(case='01')
        /path/to/gtset/01.nii.gz
        
        >>> fs.getfilename(case='01')
        01.nii.gz
        
        >>> fs.getimage(case='01')
        (img, hdr)
        
        >>> fs.getfilename(case='01', identifier='flair') # identifier argument will be ignored
        01.nii.gz
        
        >>> fs = FileSet('/path/to/traindb/', ['01', '02'], ['flair', 't1'], ['flair.nii.gz', 't1.nii.gz'], 'cases', 'identifiers')
        
        /path/to/traindb/01/flair.nii.gz
                        /01/t1.nii.gz
        /path/to/traindb/02/flair.nii.gz
                        /02/t1.nii.gz                
        
        >>> fs.getfile(case='01', identifier='flair')
        /path/to/traindb/01/flair.nii.gz
        
        >>> fs.getfile(case='01')
        UnsupportedCombinationError('You must supply "case" as well as "identifier" for the current configuration.')     
        """
        # check if all attributes are valid
        if not filesource in ['cases', 'identifiers']:
            raise ValueError('"filesource" must be one of {cases, identifiers}.')
        if subdirectories and 'cases' == filesource:
            raise ValueError('"filesource" can not be set to cases if "subdirectories" is True.')
        if not cases and not identifiers:
            raise ValueError('At least one of "cases" and "identifiers" must be not False.')
        
        if 'cases' == filesource:
            if not len(filenames) == len(cases):
                raise ValueError('With "filesource" set to "cases", the number of "filenames" must equal the number of "cases".')
            self.filenamemapping = dict(zip(cases, filenames))
        else:
            if not len(filenames) == len(identifiers):
                raise ValueError('With "filesource" set to "identifiers", the number of "filenames" must equal the number of "identifiers".')
            self.filenamemapping = dict(zip(identifiers, filenames))             
        
        # set filenames in sorted order by their identifier
        self.filenames = [fn for (_, fn) in sorted(self.filenamemapping.items())]
        
        # set instance variables
        self.directory = directory
        self.cases = list(cases) if cases else cases
        self.identifiers = list(identifiers) if identifiers else identifiers
        self.subdirectories = subdirectories
        self.filesource = filesource
        
        if self.subdirectories:
            self.filebasestring = '{directory}/{case}/{filename}'
        else:
            self.filebasestring = '{directory}/{filename}'
            
        # create missing idrectories
        mkdircond(self.directory)
        if self.subdirectories:
            for case in cases:
                mkdircond(os.path.join(self.directory, case))
        
    @staticmethod
    def fromfileset(directory, fileset):
        r"""
        Create a new file set from an existing one.
        
        Parameters
        ----------
        directory : string
            Path to the directory in which to locate the newly created file set. Created,
            if not existent.
        fileset : FileSet
            Example file set from which to copy the structure.
        """
        return FileSet(directory, fileset.cases, fileset.identifiers, fileset.filenames, fileset.filesource, fileset.subdirectories)
        
    @staticmethod
    def fromdirectory(directory, sequence, filesource='identifiers'):
        r"""
        Create a new file set from an existing structure in ``directory``.
        
        Parameters
        ----------
        directory : string
            A directory containing a file set.
        sequence : sequence of strings
            The identifiers to which the (alphabetically ordered) file should be mapped.
            Can either represent the ``cases`` or the ``identifiers``.
        filesource : {'cases', 'identifiers'}
            On which to base the files. All encountered sub-directories are assumed to be
            of the type not supplied here.
            
        Returns
        -------
        fileset : FileSet
            A new fileset representing ``directory``.
            
        Raises
        ------
        ConsistencyError
            When the parsed directory is of an invalid structure.
        """
        # check for empty directory
        if not os.path.isdir(directory):
            raise ConsistencyError('Directory "{}" does not exist'.format(directory))
        
        # read in directories on next level
        _, dirnames, _ = os.walk(directory).next()
        # if existent, take them as cases and check their contents
        if not 0 == len(dirnames):
            if 'cases' == filesource:
                raise ValueError('"filesource" can not be set to cases when there are subdirectories present.')
            cases = sorted(dirnames)
            identifiers = sequence
            subdirectories = True
            _, _, expected_filenames = os.walk(os.path.join(directory, cases[0])).next()
            expected_filenames = sorted(expected_filenames)
            for case in cases[1:]:
                _, _, filenames = os.walk(os.path.join(directory, case)).next()
                filenames = sorted(filenames)
                if not filenames == expected_filenames:
                    raise ConsistencyError('non-consistent image files in case folder "{}": expected {}, got {}'.format(os.path.join(directory, case), expected_filenames, filenames))
        
        else: # if not existent, read files from file set directory directly
                _, _, filenames = os.walk(directory).next()
                filenames = sorted(filenames)
                identifiers = sequence if 'identifiers' == filesource else False
                cases = False if identifiers else sequence
                subdirectories = False
        
        # create and return ImageSet object
        return FileSet(directory, cases, identifiers, filenames, filesource, subdirectories) 
        
    def getfiles(self, case=False, identifier=False):
        r"""
        Returns the paths to all requested files, identified by one or none of
        ``case`` and ``identifier``.
        
        In the case of deep file sets:
        
        - supplying no argument results in all files belonging to the file set
        - supplying a ``case`` leads to all files belonging to this case
        - supplying a ``identifier`` leads to all files belonging to this identifier
        
        In the case of flat file sets:
        
        -  supplying no argument results in all files belonging to the file set
        
        Parameters
        ----------
        case : False or string
            A case. 
        identifier : False or string
            An identifier. Supply only one or none of ``case`` and ``identifier``.
        
        Returns
        -------
        files : list of strings
            Paths to the requested files.
            
        Raises
        ------
        ValueError
            If case and identifier are supplied.
        UnsupportedCombinationError
            If ``case`` respectively ``identifier`` is supplied but not supported by the
            file set (i.e. a flat hierarchy).
        """
        if case and identifier:
            raise ValueError('Only one of "case" and "identifier" can be supplied.')
        if not self.subdirectories and (case or identifier):
            raise UnsupportedCombinationError('"case" and "identifier" can only be used with deep hierarchies (i.e. subdirectories=True).')
        
        if self.subdirectories:
            if case:
                return [self.getfile(case=case, identifier=identifier) for identifier in self.identifiers]
            elif identifier:
                return [self.getfile(case=case, identifier=identifier) for case in self.cases]
            else:
                return [self.getfile(case=case, identifier=identifier) for case, identifier in itertools.product(self.cases, self.identifiers)]
        else:
            if 'cases' == self.filesource:
                return [self.getfile(case=case) for case in self.cases]
            else:
                return [self.getfile(identifier=identifier) for identifier in self.identifiers]
        
    def getfile(self, case=False, identifier=False):
        r"""
        Returns the path to the requested file, identified by one or a combination of
        ``case`` and ``identifier``.
        
        Parameters
        ----------
        case : False or string
            A case.
        identifier : False or string
            An identifier. At least one of ``case`` and ``identifier`` must be supplied.
        
        Returns
        -------
        file : string
            Path to the requested file.
            
        Raises
        ------
        ValueError
            If neither case nor identifier are supplied.
        UnsupportedCombinationError
            If the provided combination of ``case`` and ``identifier`` is not supported
            by the file set.
            
        Examples
        --------
        For deep-hierarchies i.e. file sets with sub-folders enabled, you will need to
        supply both ``case`` and ``identifier`` to uniquely identify a file. E.g.
        
        >>> deepfileset.getfile(case='02', identifier='flair')
        
        returns the file belonging to case *02* and the identifier *flair*, while
        
        >>> deepfileset.getfile(case='02')
        
        results in an ``UnsupportedCombinationError`` exception.
        
        
        Flat-hierachies, on the other hand, require only one of  ``case`` and
        ``identifier`` to be supplied. E.g.
        
        >>> flatfileset.getfile(case='02')
        
        returns the file belonging to case *02*. It is furthermore possible to provide
        both parameters and let the method chose which one serves as the ``filesource`` 
        
        >>> flatfileset.getfile(case='02', identifiers='flair')
        
        returns the file belonging to case *02* for file sets with ``filesource`` set to
        *cases* and the file belonging to identifier *flair* in the case of file sets
        with ``identifiers`` set to *cases*.
        """
        if not case and not identifier:
            raise ValueError('At least one of "case" and "identifier" must be supplied.')
        if self.subdirectories and not (case and identifier):
            raise UnsupportedCombinationError('You must supply "case" as well as "identifier" for the current configuration (sub-directories enabled).')
        
        return self.filebasestring.format(filename=self.__getfilenamebynativeidentifier(case, identifier),
                                          directory=self.directory,
                                          **{'case': case, 'identifier': identifier})
        
    def __getfilenamebynativeidentifier(self, case, identifier):
        r"""
        Depending on the FileSet type, the filename is associated either with the
        ``case`` or the ``identifier``. This function allows to retrieve the filename
        without knowing the actual type of the FileSet instance.
        """
        if 'cases' == self.filesource:
            return self.filenamemapping[case]
        else:
            return self.filenamemapping[identifier]
        
        
        
        
            