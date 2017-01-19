# SearchTrees

This repository implements the nearest-neighbour search tree code and benchmarks found in my thesis, titled "Improved Similarity Search for Large Data in Machine Learning and Robotics". This code is released under the CC-BY-NC-SA Creative Commons License.

Four search tree algorithms are implemented: KD-trees; vantage-point ball trees; HDR-trees; and cover trees. These implement K-nearest-neighbour search, epsilon-nearest-neighbour search, and reverse-epsilon-nearest-neighbour search. All trees have an option for limiting the number of leaves visited, in order to perform approximate search. In addition, spill-tree variants of each tree will be made available soon.

## Usage

Each tree is templated with a data object and metric (although the KD-tree will only be effective for the Euclidean metric). The data item and metric objects are required to support the following operations:
* item.value() - return a reference to the underlying data representation of the object (usually a vector)
* item.radius() - return the sensitivity radius of the object for a reverse-nearest-neighbour query. This can be 0 for normal KNN search objects.
* item.size() - KD-tree only. This returns the length of the value() vector.
* metric::dist(item1, item2) - a static function which returns the distance between item1 and item2 as a double.

## Reproducing Thesis Results

My thesis makes use of the BigANN datasets heavily for benchmarking, since they are large and high-dimensional. The original page can be found at: http://corpus-texmex.irisa.fr

To download and format the data for reproducing the search tree results in my thesis, run:
* $ ./support/getdatasets.sh
* $ python support/reformatdatasets.py

From the directory that the formatted datasets are contained in, run:
* $ ./bin/ThesisDataSizeTests > datasizeresults.csv
* $ ./bin/ThesisLeafLimitTests > leaflimitresults.csv
* $ ./bin/ThesisSpillTreeTests > spilltreeresults.csv

Finally, to generate the figures contained in the thesis, run:
* $ python support/fig2_5.py
* $ python support/fig2_6.py
* $ python support/fig2_7.py
* $ python support/fig2_8.py
* $ python support/fig2_9.py
