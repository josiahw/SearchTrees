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

    std::pair<double, double> times;
    double accuracy = 0.;
    double stddev = 0.;

    std::vector<double> s1Vals, const1Vals;
    std::vector<size_t> leafVals;
    std::vector<std::string> scaletype;
    for (size_t j = 0; j < 60; ++j) {
        double i = j/100.;
        s1Vals.push_back(i);
        const1Vals.push_back(1.0);
        leafVals.push_back(std::numeric_limits<size_t>::max());
        scaletype.push_back("tri-gen");
    }
    for (size_t j = 6; j <= 10; ++j) {
        double i = j/10.;
        s1Vals.push_back(i);
        const1Vals.push_back(1.0);
        leafVals.push_back(std::numeric_limits<size_t>::max());
        scaletype.push_back("s-approx");
    }
    for (size_t j = 0; j < 6; ++j) {
        double i = j/10.;
        const1Vals.push_back(i);
        s1Vals.push_back(1.0);
        leafVals.push_back(std::numeric_limits<size_t>::max());
        scaletype.push_back("tri-gen");
    }
    for (size_t j = 180; j <= 300; ++j) {
        double i = j/300.;
        const1Vals.push_back(i);
        s1Vals.push_back(1.0);
        leafVals.push_back(std::numeric_limits<size_t>::max());
        scaletype.push_back("tri-gen");
    }
    for (size_t j = 0; j < 200; j += 20) {
        const1Vals.push_back(1.0);
        s1Vals.push_back(1.0);
        leafVals.push_back(j);
        scaletype.push_back("leaf-limit");
    }
    for (size_t j = 200; j < 1000; j += 50) {
        const1Vals.push_back(1.0);
        s1Vals.push_back(1.0);
        leafVals.push_back(j);
        scaletype.push_back("leaf-limit");
    }
    for (size_t j = 1000; j < 5000; j += 100) {
        const1Vals.push_back(1.0);
        s1Vals.push_back(1.0);
        leafVals.push_back(j);
        scaletype.push_back("leaf-limit");
    }

    std::cout << "APPROX VALUE, SCALE TYPE, DATASET, ACCURACY, TIME" << std::endl;

    using Test = ThesisTest<CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>>;
    const size_t testSize = 1000000;
    const size_t testQueries = 1000;

    //load the SIFT dataset
    DataSet sift = Test::mapData("sift_mmapready", 128, 1000000);
    DataSet siftq = Test::mapData("sift_queries_mmapready", 128, 10000);
    auto sifts1results = Test::statsSVals(10, testSize, testQueries, s1Vals, const1Vals, leafVals, sift, siftq);
    Test::unMapData(sift);
    Test::unMapData(siftq);

    for (size_t j = 0; j < s1Vals.size(); ++j) {
        if (scaletype[j] == "s-approx") {
            std::cout << s1Vals[j] << ", ";
        } else if (scaletype[j] == "tri-gen") {
            std::cout << const1Vals[j] << ", ";
        } else {
            std::cout << leafVals[j] << ", ";
        }
        std::cout << scaletype[j] << ", SIFT, " << sifts1results.accuracy[j] << " +/- " << sifts1results.stddev[j] << ", " << sifts1results.queryTime[j] << std::endl;
    }

    //load the GIST dataset
    sift = Test::mapData("gist_mmapready", 960, 1000000);
    siftq = Test::mapData("gist_queries_mmapready", 960, 10000);
    auto gists1results = Test::statsSVals(10, testSize, testQueries, s1Vals, const1Vals, leafVals, sift, siftq);
    Test::unMapData(sift);
    Test::unMapData(siftq);

    for (size_t j = 0; j < s1Vals.size(); ++j) {
        if (scaletype[j] == "s-approx") {
            std::cout << s1Vals[j] << ", ";
        } else if (scaletype[j] == "tri-gen") {
            std::cout << const1Vals[j] << ", ";
        } else {
            std::cout << leafVals[j] << ", ";
        }
        std::cout << scaletype[j] << ", GIST, " << gists1results.accuracy[j] << " +/- " << gists1results.stddev[j] << ", " << gists1results.queryTime[j] << std::endl;
    }

    return 0;
};
