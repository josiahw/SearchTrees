#ifndef THESIS_TEST_H
#define THESIS_TEST_H

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

#include "DataReferenceEuclideanNode.h"

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

struct DataSet {
    size_t width;
    size_t length;
    uint64_t fsize;
    int fdescriptor;
    float* fmap;
    arma::Mat<float> data;
};

struct SearchResults {
    double buildTime;
    std::vector<double> queryTime;
    std::vector<double> accuracy;
    std::vector<double> stddev;
};


template <typename Tree>
class ThesisTest {

private:
    
    

public:

    static DataSet mapData(std::string fname, size_t width, size_t length) {
        uint64_t dbsize = width * length * sizeof(float);
        DataSet ds = DataSet {width, length, dbsize};
        ds.fdescriptor = open(std::string(fname).c_str(), O_RDWR);
        ds.fmap = reinterpret_cast<float*>(mmap(NULL, ds.fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, ds.fdescriptor, 0));
        if (ds.fmap == MAP_FAILED) {
            close(ds.fdescriptor);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        ds.data = arma::Mat<float>(ds.fmap, ds.width, ds.length, false);
        return std::move(ds);
    }
    
    static void unMapData(DataSet& ds) {
        munmap(ds.fmap, ds.fsize);
        close(ds.fdescriptor);
    }

    static std::pair<double,double> timeKNN(const size_t knn,
                                            const size_t datasize,
                                            const size_t querysize,
                                            const size_t maxleaves,
                                            const double spillepsilon,
                                            const double s1,
                                            const double s2,
                                            DataSet& db,
                                            DataSet& dbq) {
        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                db.data.colptr(i), db.width
                                ));
        }
        std::random_shuffle(data.begin(),data.end());
        double buildStart = get_wall_time();
        Tree t(data, spillepsilon);
        double buildEnd = get_wall_time();

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                dbq.data.colptr(i), db.width
                                ));
        }
        double knnStart = get_wall_time();
        typename Tree::nodeReturnType tmp;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves, s1, s2);
        }
        double knnEnd = get_wall_time();
        return {buildEnd - buildStart, knnEnd - knnStart};
    }
    
    static std::pair<double,double> timeKNNInsert(const size_t knn,
                                            const size_t datasize,
                                            const size_t querysize,
                                            const size_t maxleaves,
                                            const double spillepsilon,
                                            const double s1,
                                            const double s2,
                                            DataSet& db,
                                            DataSet& dbq) {
        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        double buildStart = get_wall_time();
        data.push_back(DataReferenceEuclideanNode(
                                db.data.colptr(0), db.width
                                ));
        Tree t(data);
        for (size_t i = 1; i < datasize; ++i) {
            //add the itemm to the tree
            auto item = DataReferenceEuclideanNode(
                                db.data.colptr(i), db.width
                                );
            t.insert(item);
        }
        //std::random_shuffle(data.begin(),data.end());
        double buildEnd = get_wall_time();

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                dbq.data.colptr(i), db.width
                                ));
        }
        double knnStart = get_wall_time();
        typename Tree::nodeReturnType tmp;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves, s1, s2);
        }
        double knnEnd = get_wall_time();
        return {buildEnd - buildStart, knnEnd - knnStart};
    }
    
    static std::pair<double,double> accuracyKNN(const size_t knn,
                                            const size_t datasize,
                                            const size_t querysize,
                                            const size_t maxleaves,
                                            const double spillepsilon,
                                            const double s1,
                                            const double s2,
                                            DataSet& db,
                                            DataSet& dbq) {
        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                db.data.colptr(i), db.width
                                ));
        }
        Tree t(data, spillepsilon);

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                dbq.data.colptr(i), db.width
                                ));
        }
        typename Tree::nodeReturnType tmp;
        std::vector<double> totals;
        totals.reserve(querydata.size());
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves, s1, s2);

            //make a stack and do a linear scan
            static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                           const std::pair<double,size_t>& b) {
                return std::get<0>(a) < std::get<0>(b);
            });



            std::vector<std::pair<double,size_t>> neighbourStack;
            for (size_t i = 0; i < knn; ++i) {
                double dist = DataReferenceEuclideanNode::dist(data[i],q);
                neighbourStack.push_back(std::make_pair(dist,i));

            }
            std::make_heap(neighbourStack.begin(),
                               neighbourStack.end(),
                               neighbourCmp);
            for (size_t i = knn; i < data.size(); ++i) {
                double dist = DataReferenceEuclideanNode::dist(data[i],q);
                if (dist < std::get<0>(neighbourStack.front())) {
                    std::pop_heap(neighbourStack.begin(),
                                  neighbourStack.end(),
                                  neighbourCmp);
                    neighbourStack.back() = std::make_pair(dist,i);
                    std::push_heap(neighbourStack.begin(),
                                   neighbourStack.end(),
                                   neighbourCmp);
                }

            }
            //sort both sets of neighbours
            std::sort(neighbourStack.begin(),
                       neighbourStack.end(),
                       neighbourCmp);

            std::sort(results.begin(),
                       results.end(),
                       neighbourCmp);

            //check all children for matches
            size_t i = 0;
            size_t j = 0;
            double total = 0;
            while (j < neighbourStack.size()) {
                if (std::get<1>(results[i]) == std::get<1>(neighbourStack[j])) {
                    total += 1.0;
                    ++i;
                }
                ++j;
            }
            --j;
            while (i < results.size()) {
                if (std::get<1>(results[i]) == std::get<1>(neighbourStack[j])) {
                    total += 1.0;
                    break;
                }
                ++i;
            }
            totals.push_back(total);
        }
        arma::vec vals = arma::vec(totals);
        double mean = arma::mean(vals);
        double stddev = arma::stddev(vals);
        return {mean, stddev};
    }
    
    static SearchResults statsSVals(const size_t knn,
                                const size_t datasize,
                                const size_t querysize,
                                const std::vector<double> s1Vals,
                                const std::vector<double> s2Vals,
                                DataSet& db,
                                DataSet& dbq) {
        SearchResults results;
        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                db.data.colptr(i), db.width
                                ));
        }
        double buildStart = get_wall_time();
        Tree t(data);
        double buildEnd = get_wall_time();
        results.buildTime = buildEnd - buildStart;
        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                dbq.data.colptr(i), db.width
                                ));
        }
        
        
        for (size_t i = 0; i < s1Vals.size(); ++i) {
            const auto& s1 = s1Vals[i];
            const auto& s2 = s2Vals[i];
            typename Tree::nodeReturnType tmp;
            double knnStart = get_wall_time();
            for (auto& q : querydata) {
                auto results = t.knnQuery(q, knn, tmp, std::numeric_limits<size_t>::max(), s1, s2);
            }
            double knnEnd = get_wall_time();
            results.queryTime.push_back(knnEnd - knnStart);
            std::vector<double> totals;
            totals.reserve(querydata.size());
            for (auto& q : querydata) {
                auto results = t.knnQuery(q, knn, tmp, std::numeric_limits<size_t>::max(), s1, s2);

                //make a stack and do a linear scan
                static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                               const std::pair<double,size_t>& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });



                std::vector<std::pair<double,size_t>> neighbourStack;
                for (size_t i = 0; i < knn; ++i) {
                    double dist = DataReferenceEuclideanNode::dist(data[i],q);
                    neighbourStack.push_back(std::make_pair(dist,i));

                }
                std::make_heap(neighbourStack.begin(),
                                   neighbourStack.end(),
                                   neighbourCmp);
                for (size_t i = knn; i < data.size(); ++i) {
                    double dist = DataReferenceEuclideanNode::dist(data[i],q);
                    if (dist < std::get<0>(neighbourStack.front())) {
                        std::pop_heap(neighbourStack.begin(),
                                      neighbourStack.end(),
                                      neighbourCmp);
                        neighbourStack.back() = std::make_pair(dist,i);
                        std::push_heap(neighbourStack.begin(),
                                       neighbourStack.end(),
                                       neighbourCmp);
                    }

                }
                //sort both sets of neighbours
                std::sort(neighbourStack.begin(),
                           neighbourStack.end(),
                           neighbourCmp);

                std::sort(results.begin(),
                           results.end(),
                           neighbourCmp);

                //check all children for matches
                size_t i = 0;
                size_t j = 0;
                double total = 0;
                while (j < neighbourStack.size()) {
                    if (std::get<1>(results[i]) == std::get<1>(neighbourStack[j])) {
                        total += 1.0;
                        ++i;
                    }
                    ++j;
                }
                --j;
                while (i < results.size()) {
                    if (std::get<1>(results[i]) == std::get<1>(neighbourStack[j])) {
                        total += 1.0;
                        break;
                    }
                    ++i;
                }
                totals.push_back(total);
            }
            arma::vec vals = arma::vec(totals) / knn;
            results.accuracy.push_back(arma::mean(vals));
            results.stddev.push_back(arma::stddev(vals));
        }
        return results;
    }


    static std::pair<double,double> timeKNNSIFT(const size_t knn = 10,
                                                const size_t datasize = 1000000,
                                                const size_t querysize = 1000,
                                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                                const double spillepsilon = 0.0,
                                                const double s1 = 1.0,
                                                const double s2 = 1.0) {
        //load the SIFT dataset
        DataSet sift = ThesisTest::mapData("sift_mmapready", 128, 1000000);
        DataSet siftq = ThesisTest::mapData("sift_queries_mmapready", 128, 10000);
        
        auto results = timeKNN(knn, datasize, querysize, maxleaves, spillepsilon, s1, s2, sift, siftq);
        
        ThesisTest::unMapData(sift);
        ThesisTest::unMapData(siftq);
        return results;
    }

    static std::pair<double,double> timeKNNInsertSIFT(const size_t knn = 10,
                                                const size_t datasize = 1000000,
                                                const size_t querysize = 1000,
                                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                                const double spillepsilon = 0.0,
                                                const double s1 = 1.0,
                                                const double s2 = 1.0) {
        //load the SIFT dataset
        DataSet sift = ThesisTest::mapData("sift_mmapready", 128, 1000000);
        DataSet siftq = ThesisTest::mapData("sift_queries_mmapready", 128, 10000);

        auto results = timeKNNInsert(knn, datasize, querysize, maxleaves, spillepsilon, s1, s2, sift, siftq);
        
        ThesisTest::unMapData(sift);
        ThesisTest::unMapData(siftq);
        return results;
    }

    static std::pair<double,double> timeKNNInsertGIST(const size_t knn = 10,
                                                const size_t datasize = 1000000,
                                                const size_t querysize = 1000,
                                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                                const double spillepsilon = 0.0,
                                                const double s1 = 1.0,
                                                const double s2 = 1.0) {
        //load the GIST dataset
        DataSet gist = ThesisTest::mapData("gist_mmapready", 960, 1000000);
        DataSet gistq = ThesisTest::mapData("gist_queries_mmapready", 960, 10000);

        auto results = timeKNNInsert(knn, datasize, querysize, maxleaves, spillepsilon, s1, s2, gist, gistq);

        ThesisTest::unMapData(gist);
        ThesisTest::unMapData(gistq);
        return results;
    }

    static std::pair<double,double> timeKNNGIST(const size_t knn = 10,
                                                const size_t datasize = 1000000,
                                                const size_t querysize = 1000,
                                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                                const double spillepsilon = 0.0,
                                                const double s1 = 1.0,
                                                const double s2 = 1.0) {
        //load the GIST dataset
        DataSet gist = ThesisTest::mapData("gist_mmapready", 960, 1000000);
        DataSet gistq = ThesisTest::mapData("gist_queries_mmapready", 960, 10000);

        auto results = timeKNN(knn, datasize, querysize, maxleaves, spillepsilon, s1, s2, gist, gistq);

        ThesisTest::unMapData(gist);
        ThesisTest::unMapData(gistq);
        return results;
    }

    ///Return the k-nearest-neighbour accuracy of the tree search algorithm
    static std::pair<double,double> accuracyKNNSIFT(const size_t knn = 10,
                                const size_t datasize = 1000000,
                                const size_t querysize = 1000,
                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                const double spillepsilon = 0.0,
                                const double s1 = 1.0,
                                const double s2 = 1.0) {
        //load the SIFT dataset
        DataSet sift = ThesisTest::mapData("sift_mmapready", 128, 1000000);
        DataSet siftq = ThesisTest::mapData("sift_queries_mmapready", 128, 10000);

        auto results = accuracyKNN(knn, datasize, querysize, maxleaves, spillepsilon, s1, s2, sift, siftq);
        
        ThesisTest::unMapData(sift);
        ThesisTest::unMapData(siftq);
        return results;
    }

    ///Return the k-nearest-neighbour accuracy of the tree search algorithm
    static std::pair<double,double> accuracyKNNGIST(const size_t knn = 10,
                                const size_t datasize = 1000000,
                                const size_t querysize = 1000,
                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                const double spillepsilon = 0.0,
                                const double s1 = 1.0,
                                const double s2 = 1.0) {
        //load the GIST dataset
        DataSet gist = ThesisTest::mapData("gist_mmapready", 960, 1000000);
        DataSet gistq = ThesisTest::mapData("gist_queries_mmapready", 960, 10000);

        auto results = accuracyKNN(knn, datasize, querysize, maxleaves, spillepsilon, s1, s2, gist, gistq);

        ThesisTest::unMapData(gist);
        ThesisTest::unMapData(gistq);
        return results;
    }

};

#endif

