
#ifndef EUCLIDEAN_KNNGRAPH_NODE_H
#define EUCLIDEAN_KNNGRAPH_NODE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <sys/types.h>


class EuclideanKnnGraphNode {
    //the EuclideanKnnGraphNode is a simple example class for using the KDTree to construct incremental KNN graphs
private:
    arma::Col<double> val;
    std::vector<std::pair<double, size_t>> neighbours;
    size_t maxNeighbours;

    const static bool neighbourCmp(const std::pair<double,size_t>& a,
                             const std::pair<double,size_t>& b) {
        return std::get<0>(a) < std::get<0>(b);
    }

public:
    EuclideanKnnGraphNode(const arma::Col<double>& v, size_t k) {
        val = v;
        maxNeighbours = k;
    }

    std::vector<std::pair<double, size_t>> getNeighbours() const {
        return neighbours;
    }

    void setNeighbours(const std::vector<std::pair<double, size_t>>& n) {
        neighbours = n;
    }

    void insertNeighbour(const double& dist, const size_t& index) {
        if (neighbours.size() < maxNeighbours) {
            neighbours.push_back({dist, index});
            if (neighbours.size() > 1) {
                std::push_heap(neighbours.begin(),
                              neighbours.end(),
                              neighbourCmp);
            }
        } else if (dist < std::get<0>(neighbours.front())) {
            std::pop_heap(neighbours.begin(),
                          neighbours.end(),
                          neighbourCmp);
            neighbours.back() = {dist, index};
            std::push_heap(neighbours.begin(),
                          neighbours.end(),
                          neighbourCmp);
        }
    }

    const arma::Col<double>& value() const {
        return val;
    }

    double radius() const {
        if (neighbours.size() < maxNeighbours) {
            return std::numeric_limits<double>::max();
        }
        return std::get<0>(neighbours.front());
    }

    size_t size() const {
        return val.n_elem;
    }

    ///this is the required function implementation for the METRIC object
    const static double dist(const EuclideanKnnGraphNode& a, const EuclideanKnnGraphNode& b) {
        return arma::norm(a.value() - b.value());
    }
};

#endif