Python SVM experiment manager
=============================

Usage
-----

```
cd some_folder/
python ~/workspace/py_experiment_manager/assign_example_ids.py svmlight_formatted_example_file
python ~/workspace/py_experiment_manager/prepare_files.py svmlight_formatted_example_file.numbered 5
python ~/workspace/py_experiment_manager/run_experiments.py .
```


Assumptions
-----------

* You have a good number of examples so that each fold can have at least one example of each class.
* Your servers run a modern 64-bit Linux distribution, have ```python2``` and ```tar``` installed, and ```/tmp```, ```~/bin``` are accessible.
* Machines that run this script have ```tar``` installed.
* The master script must keep running during experiments.
