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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version d0.1
# since 2014-10-01
# status Development

# build-in module
import os
import sys

# third-party modules
import numpy
from medpy.core.logger import Logger

# own modules
from .exceptions import TaskExecutionError

# constants

# code
class TaskMachine (object):
    r"""
    A class to which tasks can be registered and then executed.
    
    !TODO: Allow for parallel execution where possible.
    """
    def __init__(self) :
        """
        """
        self.logger = Logger.getInstance()
        self.tasks = []
    
    def register(self, required_files, generated_files, callback_function, args, kwargs, description):
        r"""
        Register a new task to the processing pipeline.
        
        Parameters
        ----------
        required_files : sequence of strings
            List of files that must be present for the task to succeed.
        generated_files : sequence of strings
            List of files that must be present after the execution of the task to consider it successful.
        callback_function : function
            The function that executes the task.
        args : sequence
            Positional arguments of the ``callback_function``.
        kwargs : dict
            Keyword arguments of the ``callback_function``.
        description : string
            Short description of the task.
        """
        self.tasks.append([required_files, generated_files, callback_function, args, kwargs, description])
    
    def run(self):
        r"""
        Execute the registered tasks, then empty the task list.
        """
        ntasks = len(self.tasks)
        # for each task
        for tid, (srcs, trgs, fun, args, kwargs, desc) in enumerate(self.tasks):
            tid += 1
            # check required source files
            srcs_check = self.__check_files(srcs)
            if not numpy.all(srcs_check):
                raise TaskExecutionError('Task {}/{} ({}): Required source file(s) missing: "{}"'.format(tid, ntasks, desc, numpy.asarray(srcs)[~srcs_check]))
            # check target files
            trgs_check = self.__check_files(trgs)
            if numpy.all(trgs_check):
                self.logger.warning('Task {}/{} ({}): target files already existent; skipping task'.format(tid, ntasks, desc))
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
                raise TaskExecutionError, TaskExecutionError('Task {}/{} ({}): Execution failed. Removed partial results. Reason signaled: {}'.format(tid, ntasks, desc, e)), sys.exc_info()[2]
            # check target files
            trgs_check = self.__check_files(trgs)
            if not numpy.all(srcs_check):
                raise TaskExecutionError('Task {}/{} ({}): Execution failed to create some target files: "{}".'.format(tid, ntasks, desc, numpy.asarray(trgs)[~trgs_check]))
        self.tasks = []

    @staticmethod     
    def __check_files(files):
        r"""
        Check all files for existence, return True if yes, otherwise False.
        """
        return numpy.asarray([os.path.isfile(f) for f in files])
            
