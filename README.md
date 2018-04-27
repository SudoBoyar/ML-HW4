## ML-HW4
Multiclass fingerprint identification using SVM.


To run:

python checker.py \[path/to/training/folder/\] path/to/test_file1.bmp \[path/to/test_file2.bmp ...\]

Be default, it expects the training set to be in `training/`.
The folder containing the training set can also just be passed in as the first argument.
Then it expects one or more files to test.

e.g.

`python checker.py training_set_b/ test_set_b/*`

would train on all files in the `training_set_b` folder, and then test all files in the `test_set_b` folder.
It doesn't recurse on the test folder on its own, it needs `folder/*`.


`python checker.py test.bmp`

would train on all files in `training/` and test on `test.bmp`
