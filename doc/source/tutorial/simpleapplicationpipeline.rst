===========================
Simple application pipeline
===========================

Read in the images to segment

>>> casedb = FileSet.fromdirectory(images-directory, list-of-MRI-sequences, filesource='identifiers')

Check if the mapping of MRI sequence type to file name is correct

>>> print 'Deducted sequence to file mapping:', traindb.filenamemapping

Load a trained forest instance

>>> TrainedForest.fromdirectory('directory-with-trained-forest-instance')

Exdecute the pre-processing of the images are denoted by the forest instance, starting with unifying the MRI sequences

>>> unified = unify('/tmp/training-workingdir/00unification', casedb, fixedsequence=forestinstance.fixedsequence, targetspacing=forestinstance.workingresolution)

Compute brain masks

>>> brainmasks, _ = stripskull('/tmp/training-workingdir/02skullstrip', unified, stripsequence=forestinstance.skullstripsequence)

Correct the bias fields

>>> biascorrected = correctbiasfields('/tmp/training-workingdir/03biasfield', unified, brainmasks)

Applying intensity range models

>>> standarised = percentilemodelapplication('/tmp/training-workingdir/04intensitystd', biascorrected, brainmasks, forestinstance.getintensitymodels())

Extract features

>>> features, _, fnames = extractfeatures('/tmp/training-workingdir/05features', standarised, brainmasks)

Segment the cases

>>> segmentations, probabilities = applyforest('/tmp/training-workingdir/05segmentations', forestinstance.forest, features, brainmasks)

Post-processing segmentations

>>> postprocessed = postprocess('/tmp/training-workingdir/06postprocessed', segmentations, objectthreshold=1500)

Re-sampling segmentations, probability maps and brain masks to original space

>>> origsegmentations = resamplebyexample('directory-to-place-segmentations', postprocessed, casedb, forestinstance.fixedsequence, binary=True)
>>> origprobabilities = resamplebyexample('directory-to-place-segmentations', probabilities, casedb, forestinstance.fixedsequence)
>>> origbrainmasks = resamplebyexample('directory-to-place-segmentations/brainmasks', brainmasks, casedb, forestinstance.fixedsequence, binary=True)
    
... and it is done.

