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
# version r0.1
# since 2014-10-01
# status Development

# build-in module
import os
import sys
from multiprocessing import Pool

# third-party modules
import numpy
from medpy.core.logger import Logger

# own modules
from .exceptions import TaskExecutionError

# constants

# code
class TaskMachine (object):
    r"""
    A class to which tasks can be registered and then executed, either sequential or in parallel.
    """
    def __init__(self, multiprocessing = False, nprocesses = None) :
        r"""
        A class to which tasks can be registered and then executed, either sequential or in parallel.
        
        Parameters
        ----------
        multiprocessing : bool
            Enable/disable multiprocessing.
        nprocesses : int or None
            The number of processes to spawn. If ``None``, the number corresponds to the processor count.
        """
        self.logger = Logger.getInstance()
        self.tasks = []
        self.multiprocessing = multiprocessing
        self.nprocesses = nprocesses
    
    def register(self, required_files, generated_files, callback_function, args, kwargs, description):
        r"""
        Register a new task to the processing pipeline.
        
        Parameters
        ----------
        required_files : sequence of strings
            List of files that must be present for the task to be executed.
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
        If the ``multiprocessing`` instance variable is set, the tasks are execute in parallel.
        """
        # add an id to each task as first argument
        tasks = [[tid + 1] + task for tid, task in enumerate(self.tasks)]
        # execute tasks (multiprocessing or sequential)
        if self.multiprocessing:
            pool = Pool(self.nprocesses)
            pool.map(_runtask, tasks)
        else:
            for task in tasks:
                _runtask(task)
        # empty task list
        self.tasks = []
    
## static, module-accessible methods for parallel processing
def _runtask((tid, srcs, trgs, fun, args, kwargs, desc)):
    r"""
    Execute a single task.
    """
    # initialize logger
    logger = Logger.getInstance()
    # check required source files
    srcs_check = _check_files(srcs)
    if not numpy.all(srcs_check):
        raise TaskExecutionError('Task {} ({}): Required source file(s) missing: "{}"'.format(tid, desc, numpy.asarray(srcs)[~srcs_check]))
    # check target files
    trgs_check = _check_files(trgs)
    if numpy.all(trgs_check):
        logger.warning('Task {} ({}): All target files already existent; skipping task'.format(tid, desc))
        return
    elif numpy.any(trgs_check):
        raise TaskExecutionError('Task {} ({}): Some target file(s) already exist: "{}".'.format(tid, desc, numpy.asarray(trgs)[trgs_check]))
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
        raise TaskExecutionError, TaskExecutionError('Task {} ({}): Execution failed. Partial results removed. Reason signaled: {}'.format(tid, desc, e)), sys.exc_info()[2]
    # check target files
    trgs_check = _check_files(trgs)
    if not numpy.all(srcs_check):
        raise TaskExecutionError('Task {} ({}): Execution failed to create some target files: "{}".'.format(tid, desc, numpy.asarray(trgs)[~trgs_check]))        
   
def _check_files(files):
    r"""
    Check all files for existence, return True if yes, otherwise False.
    """
    return numpy.asarray([os.path.isfile(f) for f in files])
