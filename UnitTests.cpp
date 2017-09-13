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

//#include "InMemoryKDTree.h"
#include "KDTree.h"
#include "CoverTree.h"
#include "BallTree.h"
#include "EuclideanNode.h"
#include "TestTree.h"

int main (void) {

    std::cout << "DATASET SIZES" << std::endl;
    size_t testDims = 2;
    size_t testSize = 10000;
    size_t testKnn = 10;
    double testEnn = 0.01;
    double accuracy;
    std::vector<double> results;
    //KD-Tree tests
    std::cout << "Testing Exact KD-tree accuracy" << std::endl;
    accuracy = EuclideanTestSearch< KDTree<EuclideanNode, EuclideanNode> >::accuracyKNN(testKnn, testDims, testSize);
    std::cout << "KD-tree KNN test accuracy: " << accuracy << std::endl;
    accuracy = EuclideanTestSearch< KDTree<EuclideanNode, EuclideanNode> >::accuracyInsertKNN(testKnn, testDims, testSize);
    std::cout << "KD-tree KNN test accuracy using inserts: " << accuracy << std::endl;
    accuracy = EuclideanTestSearch< KDTree<EuclideanNode, EuclideanNode> >::accuracyENN(testEnn, testDims, testSize);
    std::cout << "KD-tree ENN test accuracy: " << accuracy << std::endl << std::endl;


    std::cout << "Number of dims: " << testDims << std::endl;
    for (size_t k = 50000; k <= 1000000; k += 50000) {
        testSize = k;
        std::cout << "Dataset size: " << testSize << std::endl;
        std::pair<double, double> times = EuclideanTestSearch< KDTree<EuclideanNode, EuclideanNode> >::timeKNN(testKnn, testDims, testSize);
        std::cout << "KD-tree built in " << std::get<0>(times) << " seconds." << std::endl;
        std::cout << "KD-tree KNN queries in " << std::get<1>(times)  << " seconds." << std::endl;
        times = EuclideanTestSearch< KDTree<EuclideanNode, EuclideanNode> >::timeInsertKNN(testKnn, testDims, testSize);
        std::cout << "KD-tree built with inserts in " << std::get<0>(times) << " seconds." << std::endl;
        std::cout << "KD-tree KNN queries using inserts in " << std::get<1>(times)  << " seconds." << std::endl;
        times = EuclideanTestSearch< KDTree<EuclideanNode, EuclideanNode> >::timeENN(testEnn, testDims, testSize);
        std::cout << "KD-tree ENN queries in " << std::get<1>(times)  << " seconds." << std::endl;
    }

    testSize = 100000;
    std::cout << "Testing KD-tree KNN-graph accuracy:" << std::endl << "Size\tAccuracy" << std::endl;
    results = EuclideanTestSearch< KDTree<EuclideanKnnGraphNode, EuclideanKnnGraphNode> >::accuracyKNNGraph(testKnn, testDims, testSize, 5000);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i+1)*5000 << "\t" << results[i] << std::endl;
    }

    std::cout << "Testing KD-tree KNN-graph time:" << std::endl << "Size\tTime" << std::endl;
    results = EuclideanTestSearch< KDTree<EuclideanKnnGraphNode, EuclideanKnnGraphNode> >::timeKNNGraph(testKnn, testDims, testSize, 5000);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i+1)*5000 << "\t" << results[i] << std::endl;
    }


    //BallTree tests
    testDims = 2;
    testSize = 10000;
    testKnn = 10;
    testEnn = 0.01;

    std::cout << "Testing Exact VP-tree accuracy" << std::endl;
    accuracy = EuclideanTestSearch< BallTree<EuclideanNode, EuclideanNode> >::accuracyKNN(testKnn, testDims, testSize);
    std::cout << "VP-tree KNN test accuracy: " << accuracy << std::endl;
    accuracy = EuclideanTestSearch< BallTree<EuclideanNode, EuclideanNode> >::accuracyInsertKNN(testKnn, testDims, testSize);
    std::cout << "VP-tree KNN test accuracy using inserts: " << accuracy << std::endl;
    accuracy = EuclideanTestSearch< BallTree<EuclideanNode, EuclideanNode> >::accuracyENN(testEnn, testDims, testSize);
    std::cout << "VP-tree ENN test accuracy: " << accuracy << std::endl << std::endl;

    std::cout << "Number of dims: " << testDims << std::endl;
    for (size_t k = 50000; k <= 1000000; k += 50000) {
        testSize = k;
        std::cout << "Dataset size: " << testSize << std::endl;
        std::pair<double, double> times = EuclideanTestSearch< BallTree<EuclideanNode, EuclideanNode> >::timeKNN(testKnn, testDims, testSize);
        std::cout << "VP-tree built in " << std::get<0>(times) << " seconds." << std::endl;
        std::cout << "VP-tree KNN queries in " << std::get<1>(times)  << " seconds." << std::endl;
        times = EuclideanTestSearch< BallTree<EuclideanNode, EuclideanNode> >::timeInsertKNN(testKnn, testDims, testSize);
        std::cout << "VP-tree built with inserts in " << std::get<0>(times) << " seconds." << std::endl;
        std::cout << "VP-tree KNN queries using inserts in " << std::get<1>(times)  << " seconds." << std::endl;
        times = EuclideanTestSearch< BallTree<EuclideanNode, EuclideanNode> >::timeENN(testEnn, testDims, testSize);
        std::cout << "VP-tree ENN queries in " << std::get<1>(times)  << " seconds." << std::endl;
    }

    testSize = 100000;
    std::cout << "Testing VP-tree KNN-graph accuracy:" << std::endl << "Size\tAccuracy" << std::endl;
    results = EuclideanTestSearch< BallTree<EuclideanKnnGraphNode, EuclideanKnnGraphNode> >::accuracyKNNGraph(testKnn, testDims, testSize, 5000);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i+1)*5000 << "\t" << results[i] << std::endl;
    }

    std::cout << "Testing VP-tree KNN-graph time:" << std::endl << "Size\tTime" << std::endl;
    results = EuclideanTestSearch< BallTree<EuclideanKnnGraphNode, EuclideanKnnGraphNode> >::timeKNNGraph(testKnn, testDims, testSize, 5000);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i+1)*5000 << "\t" << results[i] << std::endl;
    }

    //CoverTree tests
    testDims = 2;
    testSize = 10000;
    testKnn = 10;
    testEnn = 0.01;

    std::cout << "Testing Exact cover-tree accuracy" << std::endl;
    accuracy = EuclideanTestSearch< CoverTree<EuclideanNode, EuclideanNode> >::accuracyKNN(testKnn, testDims, testSize);
    std::cout << "cover-tree KNN test accuracy: " << accuracy << std::endl;
    accuracy = EuclideanTestSearch< CoverTree<EuclideanNode, EuclideanNode> >::accuracyInsertKNN(testKnn, testDims, testSize);
    std::cout << "cover-tree KNN test accuracy using inserts: " << accuracy << std::endl;
    accuracy = EuclideanTestSearch< CoverTree<EuclideanNode, EuclideanNode> >::accuracyENN(testEnn, testDims, testSize);
    std::cout << "cover-tree ENN test accuracy: " << accuracy << std::endl << std::endl;

    std::cout << "Number of dims: " << testDims << std::endl;
    for (size_t k = 50000; k <= 1000000; k += 50000) {
        testSize = k;
        std::cout << "Dataset size: " << testSize << std::endl;
        std::pair<double, double> times = EuclideanTestSearch< CoverTree<EuclideanNode, EuclideanNode> >::timeKNN(testKnn, testDims, testSize);
        std::cout << "cover-tree built in " << std::get<0>(times) << " seconds." << std::endl;
        std::cout << "cover-tree KNN queries in " << std::get<1>(times)  << " seconds." << std::endl;
        times = EuclideanTestSearch< CoverTree<EuclideanNode, EuclideanNode> >::timeInsertKNN(testKnn, testDims, testSize);
        std::cout << "cover-tree built with inserts in " << std::get<0>(times) << " seconds." << std::endl;
        std::cout << "cover-tree KNN queries using inserts in " << std::get<1>(times)  << " seconds." << std::endl;
        times = EuclideanTestSearch< CoverTree<EuclideanNode, EuclideanNode> >::timeENN(testEnn, testDims, testSize);
        std::cout << "cover-tree ENN queries in " << std::get<1>(times)  << " seconds." << std::endl;
    }

    testSize = 100000;
    std::cout << "Testing cover-tree KNN-graph accuracy:" << std::endl << "Size\tAccuracy" << std::endl;
    results = EuclideanTestSearch< CoverTree<EuclideanKnnGraphNode, EuclideanKnnGraphNode> >::accuracyKNNGraph(testKnn, testDims, testSize, 5000);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i+1)*5000 << "\t" << results[i] << std::endl;
    }

    std::cout << "Testing cover-tree KNN-graph time:" << std::endl << "Size\tTime" << std::endl;
    results = EuclideanTestSearch< CoverTree<EuclideanKnnGraphNode, EuclideanKnnGraphNode> >::timeKNNGraph(testKnn, testDims, testSize, 5000);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i+1)*5000 << "\t" << results[i] << std::endl;
    }

    return 0;
};
