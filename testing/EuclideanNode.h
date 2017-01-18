
#ifndef EUCLIDEAN_NODE_H
#define EUCLIDEAN_NODE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <sys/types.h>


class EuclideanNode {
    //the EuclideanNode is a simple example class for using the KDTree
private:
    arma::Col<double> val;
    double rad;

public:
    EuclideanNode(const arma::Col<double>& v) {
        val = v;
        rad = 0.;
    }

    EuclideanNode(const arma::Col<double>& v, const double& r) {
        val = v;
        rad = r;
    }


    const arma::Col<double>& value() const {
        return val;
    }

    double radius() const {
        return rad;
    }

    size_t size() const {
        return val.n_elem;
    }

    ///this is the required function implementation for the METRIC object
    const static double dist(const EuclideanNode& a, const EuclideanNode& b) {
        return arma::norm(a.value() - b.value());
    }
};

#endif