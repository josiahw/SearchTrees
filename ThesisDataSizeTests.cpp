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

#include "BruteForce.h"
#include "KDTree.h"
#include "CoverTree.h"
#include "BallTree.h"
#include "DataReferenceEuclideanNode.h"
#include "ThesisTest.h"

int main (void) {

    std::cout << "DATASET SIZE, TREE TYPE, SIFT TIME, GIST TIME" << std::endl;
    std::pair<double, double> times;
    for (size_t i = 50000; i <= 1000000; i += 50000) {
        std::cout << i << ", Brute Force, ";
        times = ThesisTest<
                        BruteForce<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNSIFT(100, i, 1000);
        std::cout << std::get<1>(times) << ", ";
        times = ThesisTest<
                        BruteForce<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNGIST(100, i, 1000);
        std::cout << std::get<1>(times) << std::endl;

        std::cout << i << ", KD tree, ";
        times = ThesisTest<
                        KDTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNSIFT(100, i, 1000);
        std::cout << std::get<1>(times) << ", ";
        times = ThesisTest<
                        KDTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNGIST(100, i, 1000);
        std::cout << std::get<1>(times) << std::endl;


        std::cout << i << ", VP tree, ";
        times = ThesisTest<
                        BallTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNSIFT(100, i, 1000);
        std::cout << std::get<1>(times) << ", ";
        times = ThesisTest<
                        BallTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNGIST(100, i, 1000);
        std::cout << std::get<1>(times) << std::endl;


        std::cout << i << ", Cover tree, ";
        times = ThesisTest<
                        CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNInsertSIFT(100, i, 1000);
        std::cout << std::get<1>(times) << ", ";
        times = ThesisTest<
                        CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>
                        >::timeKNNInsertGIST(100, i, 1000);
        std::cout << std::get<1>(times) << std::endl;
    }

    return 0;
};
