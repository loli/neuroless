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
# since 2014-10-02
# status Development

# build-in module

# third-party modules
from sklearn.ensemble.forest import ExtraTreesClassifier
import numpy
import multiprocessing

# own modules

# constants

# code
def trainet(trainingset, **kwargs):
    r"""
    Train an ExtraTree decision forest.
    
    Note
    ----
    This function does not use a ``TaskMachine``.
    
    Parameters
    ----------
    trainingset : FileSet
        The training set file set.
    **kwargs
        Keyword arguments to pass to the forest.
        
    Returns
    -------
    forest : ExtraTreesClassifier
        A trained forest instance.
    """
    trainingfeaturesfile = trainingset.getfile(identifier='features')
    trainingclassesfile = trainingset.getfile(identifier='classes')
    
    # loading training features
    with open(trainingfeaturesfile, 'r') as f:
        training_feature_vector = numpy.load(f)
    if 1 == training_feature_vector.ndim:
        training_feature_vector = numpy.expand_dims(training_feature_vector, -1)
    with open(trainingclassesfile , 'r') as f:
        training_class_vector = numpy.load(f)

    # prepare and train the decision forest
    forest = ExtraTreesClassifier(n_jobs=multiprocessing.cpu_count(), random_state=None, **kwargs)
    forest.fit(training_feature_vector, training_class_vector)    
            
    return forest

