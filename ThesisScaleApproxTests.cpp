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

    std::cout << "S VALUE, SCALE TYPE, SIFT TIME, SIFT ACCURACY, GIST TIME, GIST ACCURACY" << std::endl;
    std::pair<double, double> times;
    double accuracy = 0.;
    double stddev = 0.;
    
    std::vector<double> s1Vals, const1Vals;
    for (size_t j = 0; j <= 500; ++j) {
        double i = j/250.-1.;
        s1Vals.push_back(i);
        const1Vals.push_back(1.0);
    }
    
    using Test = ThesisTest<CoverTree<DataReferenceEuclideanNode, DataReferenceEuclideanNode>>;
    
    //load the SIFT dataset
    DataSet sift = Test::mapData("sift_mmapready", 128, 1000000);
    DataSet siftq = Test::mapData("sift_queries_mmapready", 128, 10000);
    
    auto sifts1results = Test::statsSVals(100, 1000000, 10000, s1Vals, const1Vals, sift, siftq);
    auto sifts2results = Test::statsSVals(100, 1000000, 10000, s1Vals, s1Vals, sift, siftq);
    
    Test::unMapData(sift);
    Test::unMapData(siftq);
    
    //load the GIST dataset
    sift = Test::mapData("gist_mmapready", 960, 1000000);
    siftq = Test::mapData("gist_queries_mmapready", 960, 10000);
    
    auto gists1results = Test::statsSVals(100, 1000000, 10000, s1Vals, const1Vals, sift, siftq);
    auto gists2results = Test::statsSVals(100, 1000000, 10000, s1Vals, s1Vals, sift, siftq);
    
    Test::unMapData(sift);
    Test::unMapData(siftq);
    
    for (size_t j = 0; j < s1Vals.size(); ++j) {
        std::cout << s1Vals[j] << ", S-APPROX, " << 
                    sifts1results.queryTime[j] << ", " << 
                    sifts1results.accuracy[j] << " +/- " << sifts1results.stddev[j] << ", " <<
                    gists1results.queryTime[j] << ", " << 
                    gists1results.accuracy[j] << " +/- " << gists1results.stddev[j] << ", " <<
                    std::endl;
        std::cout << s1Vals[j] << ", TRI-GEN, " << 
                    sifts2results.queryTime[j] << ", " << 
                    sifts2results.accuracy[j] << " +/- " << sifts2results.stddev[j] << ", " <<
                    gists2results.queryTime[j] << ", " << 
                    gists2results.accuracy[j] << " +/- " << gists2results.stddev[j] << ", " <<
                    std::endl;
    }

    return 0;
};