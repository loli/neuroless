#!/usr/bin/python

import os
from neuroless.shell import tmpdir, scp, cp, mv, smv, mkdircond, emptydircond,\
    rmdircond
from neuroless.exceptions import FileSystemOperationError

with tmpdir() as t:
    print t
    
    cp('test.txt', os.path.join(t, 'test2.txt'))
    scp('test.txt', os.path.join(t, 'test3.txt'))
    try:
        scp('test.txt', os.path.join(t, 'test2.txt'))
    except FileSystemOperationError as e:
        print e
    mv(os.path.join(t, 'test2.txt'), os.path.join(t, 'test3.txt'))
    smv(os.path.join(t, 'test3.txt'), os.path.join(t, 'test4.txt'))
    try:
        smv('test.txt', os.path.join(t, 'test4.txt'))
    except FileSystemOperationError as e:
        print e
    mkdircond(os.path.join(t, 'test'))
    
    cp('test.txt', os.path.join(t, 'test', 'test2.txt'))
    
    for r, d, f in os.walk(t):
        print r, d
        print r, f
    
    emptydircond(os.path.join(t, 'test'))
    rmdircond(os.path.join(t, 'test'))
    
with tmpdir() as t:
    print t
    raise Exception()
    
        
