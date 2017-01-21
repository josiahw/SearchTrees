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


template <typename Tree>
class ThesisTest {
public:
    static std::pair<double,double> timeKNNSIFT(const size_t knn = 10,
                                                const size_t datasize = 1000000,
                                                const size_t querysize = 1000,
                                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                                const double spillepsilon = 0.0) {
        //constant values for the SIFT dataset
        const size_t width = 128;
        const size_t dblength = 1000000;
        const size_t querylength = 10000;

        //mmap the database and query values
        int dbfd;
        float* dbfmap;
        uint64_t fsize = width * dblength * sizeof(float);
        dbfd = open(std::string("sift_mmapready").c_str(), O_RDWR);
        dbfmap = reinterpret_cast<float*>(mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, dbfd, 0));
        if (dbfmap == MAP_FAILED) {
            close(dbfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> database = arma::Mat<float>(dbfmap, width, dblength, false);

        int qfd;
        float* qfmap;
        uint64_t qfsize = width * querylength * sizeof(float);
        qfd = open(std::string("sift_queries_mmapready").c_str(), O_RDWR);
        qfmap = reinterpret_cast<float*>(mmap(NULL, qfsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, qfd, 0));
        if (qfmap == MAP_FAILED) {
            close(qfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> queries = arma::Mat<float>(qfmap, width, querylength, false);


        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                database.colptr(i), width
                                ));
        }
        std::random_shuffle(data.begin(),data.end());
        double buildStart = get_wall_time();
        Tree t(data, spillepsilon);
        double buildEnd = get_wall_time();

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                queries.colptr(i), width
                                ));
        }
        double knnStart = get_wall_time();
        typename Tree::nodeReturnType tmp;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves);
        }
        double knnEnd = get_wall_time();

        munmap(dbfmap, fsize);
        close(dbfd);
        munmap(qfmap, qfsize);
        close(qfd);
        return {buildEnd - buildStart, knnEnd - knnStart};
    }

    static std::pair<double,double> timeKNNGIST(const size_t knn = 10,
                                                const size_t datasize = 1000000,
                                                const size_t querysize = 1000,
                                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                                const double spillepsilon = 0.0) {
        //constant values for the SIFT dataset
        const size_t width = 960;
        const size_t dblength = 1000000;
        const size_t querylength = 1000;

        //mmap the database and query values
        int dbfd;
        float* dbfmap;
        uint64_t fsize = width * dblength * sizeof(float);
        dbfd = open(std::string("gist_mmapready").c_str(), O_RDWR);
        dbfmap = reinterpret_cast<float*>(mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, dbfd, 0));
        if (dbfmap == MAP_FAILED) {
            close(dbfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> database = arma::Mat<float>(dbfmap, width, dblength, false);

        int qfd;
        float* qfmap;
        uint64_t qfsize = width * querylength * sizeof(float);
        qfd = open(std::string("gist_queries_mmapready").c_str(), O_RDWR);
        qfmap = reinterpret_cast<float*>(mmap(NULL, qfsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, qfd, 0));
        if (qfmap == MAP_FAILED) {
            close(qfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> queries = arma::Mat<float>(qfmap, width, querylength, false);


        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                database.colptr(i), width
                                ));
        }
        std::random_shuffle(data.begin(),data.end());
        double buildStart = get_wall_time();
        Tree t(data, spillepsilon);
        double buildEnd = get_wall_time();

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                queries.colptr(i), width
                                ));
        }
        double knnStart = get_wall_time();
        typename Tree::nodeReturnType tmp;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves);
        }
        double knnEnd = get_wall_time();

        munmap(dbfmap, fsize);
        close(dbfd);
        munmap(qfmap, qfsize);
        close(qfd);
        return {buildEnd - buildStart, knnEnd - knnStart};
    }

    ///Return the k-nearest-neighbour accuracy of the tree search algorithm
    static double accuracyKNNSIFT(const size_t knn = 10,
                                const size_t datasize = 1000000,
                                const size_t querysize = 1000,
                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                const double spillepsilon = 0.0) {
        //constant values for the SIFT dataset
        const size_t width = 128;
        const size_t dblength = 1000000;
        const size_t querylength = 10000;

        //mmap the database and query values
        int dbfd;
        float* dbfmap;
        uint64_t fsize = width * dblength * sizeof(float);
        dbfd = open(std::string("sift_mmapready").c_str(), O_RDWR);
        dbfmap = reinterpret_cast<float*>(mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, dbfd, 0));
        if (dbfmap == MAP_FAILED) {
            close(dbfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> database = arma::Mat<float>(dbfmap, width, dblength, false);

        int qfd;
        float* qfmap;
        uint64_t qfsize = width * querylength * sizeof(float);
        qfd = open(std::string("sift_queries_mmapready").c_str(), O_RDWR);
        qfmap = reinterpret_cast<float*>(mmap(NULL, qfsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, qfd, 0));
        if (qfmap == MAP_FAILED) {
            close(qfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> queries = arma::Mat<float>(qfmap, width, querylength, false);


        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                database.colptr(i), width
                                ));
        }
        std::random_shuffle(data.begin(),data.end());
        double buildStart = get_wall_time();
        Tree t(data, spillepsilon);
        double buildEnd = get_wall_time();

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                queries.colptr(i), width
                                ));
        }
        double knnStart = get_wall_time();
        typename Tree::nodeReturnType tmp;
        double total = 0;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves);

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
        }
        munmap(dbfmap, fsize);
        close(dbfd);
        munmap(qfmap, qfsize);
        close(qfd);
        return total / querydata.size() / knn;
    }

    ///Return the k-nearest-neighbour accuracy of the tree search algorithm
    static double accuracyKNNGIST(const size_t knn = 10,
                                const size_t datasize = 1000000,
                                const size_t querysize = 1000,
                                const size_t maxleaves = std::numeric_limits<size_t>::max(),
                                const double spillepsilon = 0.0) {
        //constant values for the SIFT dataset
        const size_t width = 960;
        const size_t dblength = 1000000;
        const size_t querylength = 1000;

        //mmap the database and query values
        int dbfd;
        float* dbfmap;
        uint64_t fsize = width * dblength * sizeof(float);
        dbfd = open(std::string("sift_mmapready").c_str(), O_RDWR);
        dbfmap = reinterpret_cast<float*>(mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, dbfd, 0));
        if (dbfmap == MAP_FAILED) {
            close(dbfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> database = arma::Mat<float>(dbfmap, width, dblength, false);

        int qfd;
        float* qfmap;
        uint64_t qfsize = width * querylength * sizeof(float);
        qfd = open(std::string("sift_queries_mmapready").c_str(), O_RDWR);
        qfmap = reinterpret_cast<float*>(mmap(NULL, qfsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, qfd, 0));
        if (qfmap == MAP_FAILED) {
            close(qfd);
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        arma::Mat<float> queries = arma::Mat<float>(qfmap, width, querylength, false);


        std::vector<DataReferenceEuclideanNode> data;
        data.reserve(datasize);
        for (size_t i = 0; i < datasize; ++i) {
            data.push_back(DataReferenceEuclideanNode(
                                database.colptr(i), width
                                ));
        }
        std::random_shuffle(data.begin(),data.end());
        double buildStart = get_wall_time();
        Tree t(data, spillepsilon);
        double buildEnd = get_wall_time();

        std::vector<DataReferenceEuclideanNode> querydata;
        for (size_t i = 0; i < querysize; ++i) {
            querydata.push_back(DataReferenceEuclideanNode(
                                queries.colptr(i), width
                                ));
        }
        double knnStart = get_wall_time();
        typename Tree::nodeReturnType tmp;
        double total = 0.;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn, tmp, maxleaves);

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
        }
        munmap(dbfmap, fsize);
        close(dbfd);
        munmap(qfmap, qfsize);
        close(qfd);
        return total / querydata.size() / knn;
    }

};

#endif

