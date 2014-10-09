=================
Using the scripts
=================

NeuroLess comes with a number of Python command-line scripts with pre-made processing pipelines. They facilitate training a classifier to segment pathologies in brain MRI. The following example ist for ischemic stroke lesions.


Preparations
------------
You will need a number of training cases with associated ground truth (20+ should do). Place the images in a folder of the following layout::

    [..]/<case-id>/<sequence-id>.<ending>

e.g.::

    path-to-training-images/01/flair.nii.gz
    path-to-training-images/01/t1.nii.gz
    path-to-training-images/02/flair.nii.gz
    path-to-training-images/02/t1.nii.gz
    path-to-training-images/03/flair.nii.gz
    ...
    
The associated ground-truth is placed in its own folder::

    [..]/<case-id>.<ending>
    
e.g.::

    path-to-ground-truth-images/01.nii.gz
    path-to-ground-truth-images/02.nii.gz
    ...
    
Then you'll need a directory, where you want to place your trained decision forest classifiers, lets call it ``forests``. And finally, you'll require a working directory, in which the intermediate results are placed and which will allow you to re-start the process from any pipeline-step. Note that you will need quite some disc space, depending on the file-format and the amount of training cases (with 20 cases à 2 sequences each e.g. ~1GB).


Training
--------
Now lets commence with the training by calling::

    neuroless_train.py path-to-training-images/ path-to-ground-truth-images/ forests/mynewforest/ /tmp/workingdir/ FLAIR,T1 --fixedsequence=flair --nsamples=250000 --workingresolution=3
    
Note here that the supplied MR sequence identifiers have to be in the same order as the files in the case folders. Simply run::

    ls path-to-training-images/01/*
    
to check the native ordering. After starting the script, you'll get an additional message along the lines of::

    Deducted sequence to file mapping: {'t1': 't1.nii.gz', 'flair': 'flair.nii.gz'}
    
to confirm the order. Kill the script when the ordering is wrong and supply it correctly. (Note: since the script uses multi-processing you might need to call ``killall neuroless_train.py``)

The *neuroless_train.py* offers a wide range of settings. You can see them all when calling::

    neuroless_train.py -h

Here we just used ``--fixedsequence=flair`` to declare the sequence on which the ground-truth has been painted, ``--nsamples=250000`` to denote the number of samples we would like to use to the classifier training, and ``--workingresolution=3`` to define a common spacing for all sequences.

Now wait (sometimes quite some time), until the neuro pipeline is finished. If everything worked out, you will find a couple of files in ``forests/mynewforest/``. If something failed, try to figure out what the error message means, correct the problem and run the command again. All successfully finished steps (identified by their respective files in ``/tmp/workingdir/``) are not execute again.

In ``/tmp/workingdir/`` you can find the different intermediate results and observe the progress of the pipeline execution.


Application
-----------
Now we have a trained classifier for our case and want to apply it to some (formerly unseen) cases. These we place in the same layout as the training images::

    [..]/<case-id>/<sequence-id>.<ending>
   
and call

    neuroless_apply.py path-to-images-to-segment/ forests/ directory-to-place-the-segmentations/ /tmp/workingdirapplyication/ FLAIR,T1
    
as you can see, we need again a working directory and the sequences. Probably you have noted, that not the trained forest itself (``forests/mynewforest/``), but the parent directory is supplied. This is because the application script intelligently selects from all trained forests the best fitting for the cases at hand (according to their MRI sequences).

After termination, you'll find in ``directory-to-place-the-segmentations/`` for each case the three files::

    directory-to-place-the-segmentations/<case-id>_segmentation.nii.gz
    directory-to-place-the-segmentations/<case-id>_proabilities.nii.gz
    directory-to-place-the-segmentations/brainmasks/<case-id>.nii.gz
    
The first is the binary lesion segmentation, the second the lesion probabilities (range [0, 1]) and the third the computed brain masks.

Have fun :)

