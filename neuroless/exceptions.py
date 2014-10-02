########################################

class NeurolessException(BaseException):
    """Base class for all exceptions raised in the neuroless package."""
    pass

########################################

class ConsistencyError(NeurolessException):
    """Raised when the structure of an object is violated."""
    pass

class InvalidConfigurationError(NeurolessException):
    """Raises when an invalid configuration occured in an object."""

########################################

class FileSetExecption(NeurolessException):
    """Base class for all exceptions raised by the `FileSet` class."""
    pass

class UnsupportedCombinationError(FileSetExecption):
    """Raise whenever an unsuported combination of case and identifier is passed to a FileSet."""
    pass

########################################

class FileSystemOperationError(NeurolessException):
    """Raised when a file-system level operation failed."""
    pass


########################################

class CommandExecutionError(NeurolessException):
    """Raised when the execution of a command failed to produce the expected results."""
    
    def __init__(self, cmd, rtcode, stdout, stderr, info = ""):
        """
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
        