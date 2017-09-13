#define ARMA_64BIT_WORD
#include <armadillo>

#include <cstdint>
#include <sys/types.h>

#include <random>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <list>

#include <fcntl.h>
#include <sys/mman.h>

#include <sys/time.h>
#include <iostream>
#include <sstream>

#include "KDTree.h"
#include "CoverTree.h"
#include "BallTree.h"
#include "DataReferenceEuclideanNode.h"
#include "ThesisTest.h"

int main (void) {

    std::cout << "LEAF LIMIT, TREE TYPE, SIFT TIME, SIFT ACCURACY, GIST TIME, GIST ACCURACY" << std::endl;
    std::pair<double, double> times;
    double accuracy = 0.;
    for (size_t i = 10; i <= 1000000; i = (3*i/2)) {
        std::cout << i << ", KD tree, ";
        times = ThesisTest<
                        KDTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNSIFT(100, 1000000, 1000, i);
        accuracy = ThesisTest<
                        KDTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::accuracyKNNSIFT(100, 1000000, 1000, i).first;
        std::cout << std::get<1>(times) << ", " << accuracy << ", ";
        times = ThesisTest<
                        KDTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNGIST(100, 1000000, 1000, i);
        accuracy = ThesisTest<
                        KDTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::accuracyKNNGIST(100, 1000000, 1000, i).first;
        std::cout << std::get<1>(times) << ", " << accuracy << std::endl;


        std::cout << i << ", VP tree, ";
        times = ThesisTest<
                        BallTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNSIFT(100, 1000000, 1000, i);
        accuracy = ThesisTest<
                        BallTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::accuracyKNNSIFT(100, 1000000, 1000, i).first;
        std::cout << std::get<1>(times) << ", " << accuracy << ", ";
        times = ThesisTest<
                        BallTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNGIST(100, 1000000, 1000, i);
        accuracy = ThesisTest<
                        BallTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::accuracyKNNGIST(100, 1000000, 1000, i).first;
        std::cout << std::get<1>(times) << ", " << accuracy << std::endl;


        std::cout << i << ", Cover tree, ";
        times = ThesisTest<
                        CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNSIFT(100, 1000000, 1000, i);
        accuracy = ThesisTest<
                        CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::accuracyKNNSIFT(100, 1000000, 1000, i).first;
        std::cout << std::get<1>(times) << ", " << accuracy << ", ";
        times = ThesisTest<
                        CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNGIST(100, 1000000, 1000, i);
        accuracy = ThesisTest<
                        CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::accuracyKNNGIST(100, 1000000, 1000, i).first;
        std::cout << std::get<1>(times) << ", " << accuracy << std::endl;
    }

    return 0;
};
