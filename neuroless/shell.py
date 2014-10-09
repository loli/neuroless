"""
==================================================
Shell execution functions (:mod:`neuroless.shell`)
==================================================
.. currentmodule:: neuroless.shell

This package contains various function to use shell-commands in a secure manner.

.. module:: neuroless.shell
.. autosummary::
    :toctree: generated/
    
    tmpdir
    call
    cp
    scp
    mv
    smv
    mkdircond
    emptydircond
    rmdircond
    
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
# since 2014-09-25
# status Development

# build-in module
from contextlib import contextmanager
from subprocess import PIPE, Popen
import tempfile
import os

# third-party modules
from scipy.misc import doccer

# own modules
from .exceptions import FileSystemOperationError, CommandExecutionError

# constants

# documentation templates
_src_doc = \
"""src : string
    Source file.
"""
_dest_doc = \
"""dest : string
    Destination file.
"""
_fsoe_exc_doc = \
"""FileSystemOperationError
    When the conditions for the operation are not met.
"""
_cee_exc_doc = \
"""CommandExecutionError
    When the operation failed.
"""

docdict = {'src' : _src_doc,
           'dest' : _dest_doc,
           'soe_exc' : _fsoe_exc_doc,
           'cee_exc' : _cee_exc_doc}
docfiller = doccer.filldoc(docdict)

# code
def call(args):
    r"""
    Executes the command contained in ``args``.

    Parameters
    ----------
    args : sequence of strings
        First element of ``args`` is treated as the command to execute, all others as its
        space-separated arguments.
    
    Returns
    -------
    rtcode : integer
        The return code received.
    stdout : string
        The stdout buffer.
    stderr : string
        The stderr buffer.
        
    Examples
    --------
    >>> call(['mv' '-v' 'src.txt' 'trg.txt'])
    
    Will result in the execution of ``mv -v src.txt trg.txt``.
    """
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    rtcode = p.returncode
    return rtcode, stdout, stderr

def cp(src, dest):
    r"""
    Copy a file from ``src`` to ``dest``, overriding ``dest`` if it already exist.
    
    Parameters
    ----------
    %(src)s
    %(dest)s
    
    Raises
    ------        
    %(soe_exc)s
    %(cee_exc)s
    
    See Also
    --------
    scp : non-overriding copy
    """
    if not os.path.isfile(src):
        raise FileSystemOperationError('The source file "{}" does not exist.'.format(src))
    cmd = ['cp', '-p', src, dest]
    rtcode, stdout, stderr = call(cmd)
    if not os.path.isfile(dest):
        raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Destination file not created.')

def scp(src, dest):
    r"""
    Secure-copy a file from ``src`` to ``dest``, only if ``dest`` does not already exist.
    
    Parameters
    ----------
    %(src)s
    %(dest)s
    
    Raises
    ------        
    %(soe_exc)s
    %(cee_exc)s
    
    See Also
    --------
    cp : overriding copy    
    """
    if os.path.exists(dest):
        raise FileSystemOperationError('The destination "{}" already exists.'.format(dest))
    cp(src, dest)

def mv(src, dest):
    r"""
    Move a file from ``src`` to ``dest``, overriding ``dest`` if it already exist.
    
    Parameters
    ----------
    %(src)s
    %(dest)s
    
    Raises
    ------        
    %(soe_exc)s
    %(cee_exc)s
    
    See Also
    --------
    smv : non-overriding move        
    """
    if not os.path.isfile(src):
        raise FileSystemOperationError('The source file "{}" does not exist.'.format(src))
    cmd = ['mv', src, dest]
    rtcode, stdout, stderr = call(cmd)
    if not os.path.isfile(dest):
        raise CommandExecutionError(cmd, rtcode, stdout, stderr, 'Destination file not created.')    
        
def smv(src, dest):
    r"""
    Secure-move a file from ``src`` to ``dest``, only if ``dest`` does not already exist.
    
    Parameters
    ----------
    %(src)s
    %(dest)s
    
    Raises
    ------        
    %(soe_exc)s
    %(cee_exc)s
    
    See Also
    --------
    smv : overriding move
    """
    if os.path.exists(dest):
        raise FileSystemOperationError('The destination "{}" already exists.'.format(dest))
    mv(src, dest)       

def mkdircond(directory):
    r"""
    Create a directory. If already existent, silently skip.
    
    Parameters
    ----------
    directory : string
        Path to a directory, existent or not.
    
    Raises
    ------
    OSError
        When the operation failed.
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)

def emptydircond(directory):
    r"""
    Remove all files in ``directory``.
    
    Parameters
    ----------
    directory : string
        Path to an existing directory.
        
    Raises
    ------
    OSError
        When the operation failed.
    """
    _, _, files = os.walk(directory).next()
    for _file in files:
        os.remove(os.path.join(directory, _file))

def rmdircond(directory):
    r"""
    Remove an empty directory. If not existent, silently skip.
    
    Parameters
    ----------
    directory : string
        Path to a directory, existent or not.
        
    Raises
    ------
    OSError
        When the operation failed.
    """
    if os.path.isdir(directory):
        os.rmdir(directory)

@contextmanager
def tmpdir():
    r"""
    Creates an (empty) temporary directory available and takes care of the clean-up
    afterwards.
    
    Examples
    --------
    >>> with tmpdir() as t:
    >>>    write_file_to(t)
    >>>    modify_file_in(t)
    >>>    read_file_in(t)
    
    """  
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        emptydircond(tmpdir)
        rmdircond(tmpdir)