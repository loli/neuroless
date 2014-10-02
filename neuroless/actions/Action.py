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
from abc import ABCMeta, abstractmethod, abstractproperty
import multiprocessing
import os

# third-party modules
import numpy
from medpy.core.logger import Logger

# own modules
from ..exceptions import TaskExecutionError, ActionImplementationError
import sys

# constants

# code
class Action (object):
    r"""
    Base class and common interface for pipeline action.
        
    Besides providing a common interface to execute pipeline actions and to link them
    together, this class also provides some basic functionalities that can be used by the
    action implemented in the realizations.
    
    Every action takes an `imageset` object, performs its image manipulation and returns
    a new `imageset` object giving access to the created images.
    """
    
    PREFERRED_FILE_SUFFIX = 'nii.gz'
    """The preferred medciacl image file suffix, wherever a new file has to be created."""
    
    __metaclass__ = ABCMeta
    """Mark this class as an abstract base class."""
    
    def __init__(self, cwd) :
        """
        Every child class has to call: super(<child-class-name>, self).__init__()
        
        Parameters
        ----------
        cwd : string
            The current working directory to be used by all actions.
        """
        self.logger = Logger.getInstance()
        self.processors = multiprocessing.cpu_count()
        self.cwd = cwd
    
    @abstractproperty
    def cawd(self):
        """
        The current action working directory in which to execute all actions.
        """
        pass
    
    @abstractmethod
    def getimageset(self):
        """
        Returns the output image set.
        """
        pass
    
    @abstractmethod
    def _preprocess(self):
        """
        Execute the preparations and checks required before starting the actual action.
        
        - Set tasks in ``tasks`` attribute in the form [source_files, target_files, function, args-list, kwargs-dict, description-string]
        """
        pass
    
    @abstractmethod
    def _postprocess(self):
        """
        Execute the cleaning-up operations and result checks after finishing the actual action.
        """
        pass
    
    def run(self):
        """
        Actual method called to run the action.
        """
        self._preprocess()
        self.__process_tasks()
        self._postprocess()
        
    def __process_tasks(self):
        """
        Processes all registered tasks.
        """
        if not hasattr(self, 'tasks'):
            raise ActionImplementationError('The "preprocess()" method must set the "tasks" instance attribute.')
        ntasks = len(self.tasks)
        for tid, (srcs, trgs, fun, args, kwargs, desc) in enumerate(self.tasks):
            tid += 1
            # check required source files
            srcs_check = self.__check_files(srcs)
            if not numpy.all(srcs_check):
                raise TaskExecutionError('Task {}/{} ({}): Required source file(s) missing: "{}"'.format(tid, ntasks, desc, numpy.asarray(srcs)[~srcs_check]))
            # check target files
            trgs_check = self.__check_files(trgs)
            if numpy.all(trgs_check):
                self.logger.warning('({}): Task {}/{} ({}): target files already existent; skipping task'.format(self.__class__.__name__, tid, ntasks, desc))
                continue
            elif numpy.any(trgs_check):
                raise TaskExecutionError('Task {}/{} ({}): Some target file(s) already exist: "{}".'.format(tid, ntasks, desc, numpy.asarray(trgs)[trgs_check]))
            # execute task
            try:
                fun(*args, **kwargs)
            except Exception as e:
                # remove target files (if partially created)
                for trg in trgs:
                    try:
                        if os.path.isfile(trg):
                            os.remove(trg)
                    except Exception as e:
                        pass
                e.args += ('(Note: Removed eventually partially produced task target files before re-raising this exception.)', ) 
                raise TaskExecutionError, TaskExecutionError('Task {}/{} ({}): Execution failed. Removed partial results. Reason signaled: {}'.format(tid, ntasks, desc, e)), sys.exc_info()[2]
            # check target files
            trgs_check = self.__check_files(trgs)
            if not numpy.all(srcs_check):
                raise TaskExecutionError('Task {}/{} ({}): Execution failed to create some target files: "{}".'.format(tid, ntasks, desc, numpy.asarray(trgs)[~trgs_check]))

    @staticmethod     
    def __check_files(files):
        """
        Check all files for existence, return True if yes, otherwise False.
        """
        return numpy.asarray([os.path.isfile(f) for f in files])
            
