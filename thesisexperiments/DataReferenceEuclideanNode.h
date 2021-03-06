
#ifndef DATA_REFERENCE_EUCLIDEAN_NODE_H
#define DATA_REFERENCE_EUCLIDEAN_NODE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <sys/types.h>


class DataReferenceEuclideanNode {
    //this extends the EuclideanNode class to use a reference to the vector value of the data item.
    //This extension allows the use of a mmapped file to back data arrays, allowing out of core search.
private:
    arma::Col<float> val;
    double rad;
    size_t ind = 0;
    const static bool copyData = false;

public:
    static size_t distChecks;

    DataReferenceEuclideanNode(float* v, const size_t& width) : val(v, width, copyData) {
        static size_t indexctr = 0;
        rad = 0.;
        ind = indexctr;
        indexctr++;
    }

    DataReferenceEuclideanNode(float* v, const size_t& width, const double& r) : val(v, width, copyData) {
        static size_t indexctr = 0;
        rad = r;
        ind = indexctr;
        indexctr++;
    }

    DataReferenceEuclideanNode(float* v, const size_t& width, const double& r, const size_t& index) : val(v, width, copyData) {
        rad = r;
        ind = ind;
    }


    const arma::Col<float>& value() const {
        return val;
    }

    double radius() const {
        return rad;
    }

    size_t size() const {
        return val.n_elem;
    }

    size_t index() const {
        return ind;
    }

    ///this is the required function implementation for the METRIC object
    const static double dist(const DataReferenceEuclideanNode& a, const DataReferenceEuclideanNode& b) {
        ++DataReferenceEuclideanNode::distChecks;
        return arma::norm(a.value() - b.value());
    }

    static double checks() {
        double d = DataReferenceEuclideanNode::distChecks;
        DataReferenceEuclideanNode::distChecks = 0;
        return d;
    }
};
size_t DataReferenceEuclideanNode::distChecks = 0;
#endif
