
#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <sys/types.h>


template <typename OBJ, typename METRIC>
class BruteForce {

private:
    std::vector<OBJ>& data;

public:
    typedef size_t nodeReturnType;

    ///dummy creation for brute force search
    BruteForce(std::vector<OBJ>& d,
            const double& spillEps = 0.0,
            const size_t& mLeafSize = 100) : data(d) { }

    ///dummy radius update for brute force search
    void updateRadius(const size_t& dataIndex, const double& oldRadius) {
        //TODO: make this store the largest radius
    }

    ///do a KNN query for val, using brute force
    std::vector<std::pair<double,size_t>> knnQuery(const OBJ& val,
                                                    const size_t& kneighbours,
                                                    size_t& homeNode,
                                                    const size_t& maxLeaves = std::numeric_limits<size_t>::max(),
                                                    const double& s1 = 1.0,
                                                    const double& s2 = 1.0) const {

        static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                       const std::pair<double,size_t>& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        std::vector<std::pair<double,size_t>> neighbourStack;
        for (size_t i = 0; i < kneighbours; ++i) {
            double dist = METRIC::dist(data[i],val);
            neighbourStack.push_back(std::make_pair(dist,i));

        }
        std::make_heap(neighbourStack.begin(),
                           neighbourStack.end(),
                           neighbourCmp);
        for (size_t i = kneighbours; i < data.size(); ++i) {
            double dist = METRIC::dist(data[i],val);
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
        return neighbourStack;
    };

    std::vector<std::pair<double,size_t>> knnQuery(const OBJ& val,
                                                    const size_t& kneighbours) {
        size_t homeNodeRef;
        return knnQuery(val, kneighbours, homeNodeRef);
    }

    ///do a reverse-ENN query for val, with an optional leaf limit
    std::vector<std::pair<double,size_t>> rennQuery(const OBJ& val,
                                                    const size_t& maxLeaves=800000) const {
        std::vector<std::pair<double,size_t>> neighbourStack;

        for (size_t i = 0; i < data.size(); ++i) {
            double dist = METRIC::dist(data[i],val);
            if (dist < data[i].radius()) {
                neighbourStack.push_back({dist, i});
            }
        }

        return neighbourStack;
    };

    ///do a epsilon-NN query for val, with an optional leaf limit
    std::vector<std::pair<double,size_t>> ennQuery(const OBJ& val,
                                                    const double& eps,
                                                    const size_t& maxLeaves=800000) const {

        std::vector<std::pair<double,size_t>> neighbourStack;

        for (size_t i = 0; i < data.size(); ++i) {
            double dist = METRIC::dist(data[i],val);
            if (dist < eps) {
                neighbourStack.push_back({dist, i});
            }
        }

        return neighbourStack;
    };

    //insert takes an object and an optional leaf node index and inserts the object into the tree
    void insert(OBJ& item, size_t node = std::numeric_limits<size_t>::max()) {
        data.push_back(item);

        //update the tree constraints
        updateRadius(data.size()-1, 0.0);
    }

    //TODO: maybe delete and knn-renn query? Optional at this stage.
    //TODO2: fix distance estimation to be more awesome.
};

#endif

