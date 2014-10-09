========================
Simple training pipeline
========================

Read in the training images

>>> traindb = FileSet.fromdirectory(training-images-directory, list-of-MRI-sequences, filesource='identifiers')

Check if the mapping of MRI sequence type to file name is correct

>>> print 'Deducted sequence to file mapping:', traindb.filenamemapping

Read in the ground-truth images for each case

>>> gtset = FileSet.fromdirectory(ground-truth-directory, traindb.cases, filesource='cases')

Unifying MRI sequences

>>> unified = unify('/tmp/training-workingdir/00unification', traindb, fixedsequence='flair')

Re-sample the ground-truth accordingly

>>> gtunified = resample('/tmp/training-workingdir/01gtunification', gtset, order=1)

Compute brain masks

>>> brainmasks, stripsequence = stripskull('/tmp/training-workingdir/02skullstrip', unified)

Correct the bias fields

>>> biascorrected = correctbiasfields('/tmp/training-workingdir/03biasfield', unified, brainmasks)

Computing and applying intensity range models

>>> standarised, intstdmodels = percentilemodelstandardisation('/tmp/training-workingdir/04intensitystd', biascorrected, brainmasks)

Extract features

>>> features, classes, fnames = extractfeatures('/tmp/training-workingdir/05features', standarised, brainmasks, gtunified)

Sampling training-set

>>> trainingset, samplepointset = sample('/tmp/training-workingdir/06samplingset', features, classes, brainmasks, sampler='stratifiedrandomsampling', nsamples=100000)

Training decision forest

>>> forest = trainet(trainingset, n_estimators = 200)

Creating and setting forest instance

>>> forestinstance = TrainedForest('directory-to-place-forest', list-of-MRI-sequences)
>>> forestinstance.forest = forest
>>> forestinstance.trainingimages = traindb
>>> forestinstance.samplingparameters = ['stratifiedrandomsampling', 100000]
>>> forestinstance.forestparameters = {'n_estimators': 200}
>>> forestinstance.fixedsequence = 'flair'
>>> forestinstance.workingresolution = 1
>>> forestinstance.skullstripsequence = stripsequence
>>> for sequence in intstdmodels.identifiers:
        model = intstdmodels.getfile(identifier=sequence)
        with open(model, 'rb') as f:
            forestinstance.setintensitystdmodel(sequence, pickle.load(f))
        
Persist forest instance

>>> forestinstance.persist()

... and it is done.

