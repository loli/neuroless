"""
========================================
Exceptions (:mod:`neuroless.exceptions`)
========================================
.. currentmodule:: neuroless.exceptions

Custom exceptions used by the NeuroLess package. 

.. module:: neuroless.exceptions
.. autosummary::
    :toctree: generated/
    
    NeurolessException
    ConsistencyError
    InvalidConfigurationError
    TaskExecutionError
    FileSetExecption
    UnsupportedCombinationError
    FileSystemOperationError
    CommandExecutionError
    
"""
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
# since 2014-10-02
# status Development

# build-in module

# third-party modules

# own modules

# constants

# code

########################################

class NeurolessException(BaseException):
    r"""Base class for all exceptions raised in the neuroless package."""
    pass

########################################

class ConsistencyError(NeurolessException):
    r"""Raised when the structure of an object is violated."""
    pass

class InvalidConfigurationError(NeurolessException):
    r"""Raises when an invalid configuration occured in an object."""
    
class TaskExecutionError(NeurolessException):
    r"""Raises when The execution of a task failed."""

########################################

class FileSetExecption(NeurolessException):
    r"""Base class for all exceptions raised by the `FileSet` class."""
    pass

class UnsupportedCombinationError(FileSetExecption):
    r"""Raise whenever an unsuported combination of case and identifier is passed to a FileSet."""
    pass

########################################

class FileSystemOperationError(NeurolessException):
    r"""Raised when a file-system level operation failed."""
    pass

########################################

class CommandExecutionError(NeurolessException):
    r"""Raised when the execution of a command failed to produce the expected results."""
    
    def __init__(self, cmd, rtcode, stdout, stderr, info = ""):
        r"""
        Parameters
        ----------
        cmd : sequence of strings
            The command execute as sequence of strings.
        rtcode : integer
            The return-code from the command execution.
        stdout : string
            The STDOUT message.
        stderr : string
            The STDERR message
        info : string
            Additional information string describing the error in more detail.
        """
        message = """
        Running "{}" did not produce the expected results: {}
        Return-code:\t{}
        Stdout:
        -------
        {}
        -------
        Stderr:
        -------
        {}
        -------
        """.format(' '.join(cmd), info, rtcode, stdout, stderr)
        super(CommandExecutionError, self).__init__(message)
        