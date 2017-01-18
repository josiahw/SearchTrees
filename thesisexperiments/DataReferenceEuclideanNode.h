
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

public:
    DataReferenceEuclideanNode(float* v, const size_t& width) : val(v, width, false) {
        rad = 0.;
    }

    DataReferenceEuclideanNode(float* v, const size_t& width, const double& r) : val(v, width, false) {
        rad = r;
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

    ///this is the required function implementation for the METRIC object
    const static double dist(const DataReferenceEuclideanNode& a, const DataReferenceEuclideanNode& b) {
        return arma::norm(a.value() - b.value());
    }
};

#endif