We are not aware of any Python version this doesn't work with, but we tested with Python 3.9 under Windows 10.
No special hardware is required beyond standard memory requirements for scientific computing.
Required Python libraries are numpy, Pillow, opencv, scipy, kimimaro, wknml, scikit-learn
These should install within minutes once the environment is set up.

main.py has a runtime of less than a minute
The first part of matcher.py runs for several hours, but we have supplied the output of that calculation in this repository. The rest of the file runs within 10 minutes, and faster if the NML output is disabled.
The runtime considerations of repair.py are similar to those of matcher.py

Each of the three python files should run independently and can be run out of a standard Python 3 environment, except that repair.py either needs the cached version or the fresh output of the first part of matcher.py.
No further installation is necessary.

main.py outputs the estimated gap size between the pillars.
matcher.py connects skeletons across a gap and outputs NML files for manual review.
repair.py uses heuristics to fix splits in the segmentation of myelinated axons and outputs NML files for manual review.