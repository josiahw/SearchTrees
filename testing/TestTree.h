#ifndef TEST_TREE_H
#define TEST_TREE_H

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

#include "EuclideanNode.h"
#include "EuclideanKNNGraphNode.h"

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


template <typename Tree>
class EuclideanTestSearch {
public:
    static std::pair<double,double> timeKNN(const size_t knn = 10, const size_t dim = 2, const size_t size = 100000, const size_t queries = 5000) {
        std::vector<EuclideanNode> data;
        data.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            data.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double buildStart = get_wall_time();
        Tree t(data);
        double buildEnd = get_wall_time();

        std::vector<EuclideanNode> querydata;
        for (size_t i = 0; i < queries; ++i) {
            querydata.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double knnStart = get_wall_time();
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn);
        }
        double knnEnd = get_wall_time();

        return {buildEnd - buildStart, knnEnd - knnStart};
    }

    static std::pair<double,double> timeInsertKNN(const size_t knn = 10, const size_t dim = 2, const size_t size = 100000, const size_t queries = 5000) {
        std::vector<EuclideanNode> data;
        data.reserve(size);
        double buildStart = get_wall_time();
        //create the initial data
        data.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        Tree t(data);
        //generate and insert the next reportFreq items into the graph
        for (size_t i = 1; i < size; ++i) {

            //create the new data item
            auto item = EuclideanNode(arma::Col<double>(dim,arma::fill::randu));

            //add the itemm to the tree
            t.insert(item);
        }
        double buildEnd = get_wall_time();

        std::vector<EuclideanNode> querydata;
        for (size_t i = 0; i < queries; ++i) {
            querydata.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double knnStart = get_wall_time();
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn);
        }
        double knnEnd = get_wall_time();

        return {buildEnd - buildStart, knnEnd - knnStart};
    }

    ///Run a timed epsilon-nearest-neighbour test and report results
    static std::pair<double,double> timeENN(const double enn = 0.01, const size_t dim = 2, const size_t size = 100000, const size_t queries = 5000) {
        std::vector<EuclideanNode> data;
        data.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            data.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double buildStart = get_wall_time();
        Tree t(data);
        double buildEnd = get_wall_time();

        std::vector<EuclideanNode> querydata;
        for (size_t i = 0; i < queries; ++i) {
            querydata.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double ennStart = get_wall_time();
        for (auto& q : querydata) {
            auto results = t.ennQuery(q, enn);
        }
        double ennEnd = get_wall_time();

        return {buildEnd - buildStart, ennEnd - ennStart};
    }

    //TODO: implement this
    static std::vector<double> timeKNNGraph(const size_t knn = 10, const size_t dim = 2, const size_t size = 100000, const size_t reportFreq = 5000) {
        std::vector<EuclideanKnnGraphNode> data;
        std::vector<double> times;
        data.reserve(size);

        double knnStart = get_wall_time();
        //create the initial data
        data.push_back(EuclideanKnnGraphNode(arma::Col<double>(dim,arma::fill::randu), knn));
        Tree t(data);
        for (size_t j = 0; j < size/reportFreq; ++j) {
            //generate and insert the next reportFreq items into the graph
            for (size_t i = std::max(size_t(1),j*reportFreq); i < (j+1)*reportFreq; ++i) {

                //create the new data item
                EuclideanKnnGraphNode item = EuclideanKnnGraphNode(arma::Col<double>(dim,arma::fill::randu), knn);

                //set the knn's for the item
                typename Tree::nodeReturnType insertNode;
                item.setNeighbours(t.knnQuery(item,knn,insertNode));

                auto renn = t.rennQuery(item);

                //set the reverse-knn's for other items
                for (const auto& p : renn) {
                    double old = data[std::get<1>(p)].radius();
                    data[std::get<1>(p)].insertNeighbour(std::get<0>(p),data.size());
                    t.updateRadius(std::get<1>(p),old);
                }

                //add the itemm to the tree
                t.insert(item,insertNode);
            }
            times.push_back(get_wall_time() - knnStart);
        }
        return times;
    }

    ///Return the k-nearest-neighbour accuracy of the tree search algorithm
    static double accuracyKNN(const size_t knn = 10, const size_t dim = 2, const size_t size = 100000, const size_t queries = 5000) {
        std::vector<EuclideanNode> data;
        data.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            data.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        Tree t(data);

        std::vector<EuclideanNode> querydata;
        for (size_t i = 0; i < queries; ++i) {
            querydata.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double total = 0.;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn);

            /*
            make a stack and do a linear scan
            */
            static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                           const std::pair<double,size_t>& b) {
                return std::get<0>(a) < std::get<0>(b);
            });



            std::vector<std::pair<double,size_t>> neighbourStack;
            for (size_t i = 0; i < knn; ++i) {
                double dist = EuclideanNode::dist(data[i],q);
                neighbourStack.push_back(std::make_pair(dist,i));

            }
            std::make_heap(neighbourStack.begin(),
                               neighbourStack.end(),
                               neighbourCmp);
            for (size_t i = knn; i < data.size(); ++i) {
                double dist = EuclideanNode::dist(data[i],q);
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
        return total / querydata.size() / knn;
    }

    ///Return the k-nearest-neighbour accuracy of the tree search algorithm using inserts
    static double accuracyInsertKNN(const size_t knn = 10, const size_t dim = 2, const size_t size = 100000, const size_t queries = 5000) {
        std::vector<EuclideanNode> data;
        data.reserve(size);
        //create the initial data
        data.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        Tree t(data);
        //generate and insert the next reportFreq items into the graph
        for (size_t i = 1; i < size; ++i) {

            //create the new data item
            auto item = EuclideanNode(arma::Col<double>(dim,arma::fill::randu));

            //add the itemm to the tree
            t.insert(item);
        }

        std::vector<EuclideanNode> querydata;
        for (size_t i = 0; i < queries; ++i) {
            querydata.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double total = 0.;
        for (auto& q : querydata) {
            auto results = t.knnQuery(q, knn);

            /*
            make a stack and do a linear scan
            */
            static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                           const std::pair<double,size_t>& b) {
                return std::get<0>(a) < std::get<0>(b);
            });



            std::vector<std::pair<double,size_t>> neighbourStack;
            for (size_t i = 0; i < knn; ++i) {
                double dist = EuclideanNode::dist(data[i],q);
                neighbourStack.push_back(std::make_pair(dist,i));

            }
            std::make_heap(neighbourStack.begin(),
                               neighbourStack.end(),
                               neighbourCmp);
            for (size_t i = knn; i < data.size(); ++i) {
                double dist = EuclideanNode::dist(data[i],q);
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
        return total / querydata.size() / knn;
    }

    ///Check the accuracy of epsilon-nearest neighbours queries.
    static double accuracyENN(const double enn = 10, const size_t dim = 2, const size_t size = 100000, const size_t queries = 5000) {
        std::vector<EuclideanNode> data;
        data.reserve(size);
        size_t queryTotal = 0;
        for (size_t i = 0; i < size; ++i) {
            data.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        Tree t(data);

        std::vector<EuclideanNode> querydata;
        for (size_t i = 0; i < queries; ++i) {
            querydata.push_back(EuclideanNode(arma::Col<double>(dim,arma::fill::randu)));
        }
        double total = 0.;
        for (auto& q : querydata) {
            auto results = t.ennQuery(q, enn);
            /*
            make a stack and do a linear scan
            */
            static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                           const std::pair<double,size_t>& b) {
                return std::get<0>(a) < std::get<0>(b);
            });



            std::vector<std::pair<double,size_t>> neighbourStack;
            for (size_t i = 0; i < data.size(); ++i) {
                double dist = EuclideanNode::dist(data[i],q);
                if (dist <= enn) {
                    neighbourStack.push_back(std::make_pair(dist,i));
                }
            }
            queryTotal += std::max(neighbourStack.size(), results.size());
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
        return total / queryTotal;
    }

    ///TODO: implement this
    static std::vector<double> accuracyKNNGraph(const size_t knn = 10, const size_t dim = 2, const size_t size = 100000, const size_t reportFreq = 5000) {
        std::vector<EuclideanKnnGraphNode> data;
        std::vector<double> accuracies;
        data.reserve(size);


        //create the initial data
        data.push_back(EuclideanKnnGraphNode(arma::Col<double>(dim,arma::fill::randu), knn));
        Tree t(data);
        for (size_t j = 0; j < size/reportFreq; ++j) {
            //generate and insert the next reportFreq items into the graph
            for (size_t i = std::max(size_t(1),j*reportFreq); i < (j+1)*reportFreq; ++i) {
                //std::cout << "querying" << std::endl;
                //create the new data item
                EuclideanKnnGraphNode item = EuclideanKnnGraphNode(arma::Col<double>(dim, arma::fill::randu), knn);
                //set the knn's for the item
                //std::cout << "setting neighbours" << std::endl;
                item.setNeighbours(t.knnQuery(item,knn));

                //set the reverse-knn's for other items
                //std::cout << "reverse querying" << std::endl;
                auto renn = t.rennQuery(item);

                for (const auto& p : renn) {
                    double old = data[p.second].radius();
                    //std::cout << "inserting neighbours" << std::endl;
                    data[p.second].insertNeighbour(p.first,data.size());
                    //std::cout << "updating radius" << std::endl;
                    t.updateRadius(p.second,old);
                }

                //add the itemm to the tree
                //std::cout << "inserting item" << std::endl;
                t.insert(item);
            }

            //accuracy checks
            double total = 0.0;
            for (size_t i = 0; i < reportFreq/10; ++i) {

                static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                           const std::pair<double,size_t>& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });

                auto results = t.knnQuery(data[i],knn+1);
                auto neighbourStack = data[i].getNeighbours();

                //sort both sets of neighbours
                std::sort(neighbourStack.begin(),
                           neighbourStack.end(),
                           neighbourCmp);


                std::sort(results.begin(),
                           results.end(),
                           neighbourCmp);
                //remove self
                results.erase(results.begin(),results.begin()+1);

                //check all children for matches

                size_t k = 0;
                size_t l = 0;
                while (k < neighbourStack.size()) {
                    if (std::get<1>(results[l]) == std::get<1>(neighbourStack[k])) {
                        total += 1.0;
                        ++l;
                    }
                    ++k;
                }
                --k;
                while (l < results.size()) {
                    if (std::get<1>(results[l]) == std::get<1>(neighbourStack[k])) {
                        total += 1.0;
                        break;
                    }
                    ++l;
                }

            }
            accuracies.push_back(total/reportFreq*10/knn);
        }
        return accuracies;
    }
};

#endif

