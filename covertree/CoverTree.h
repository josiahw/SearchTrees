
#ifndef COVERTREE_H
#define COVERTREE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>
#include <sys/types.h>

#include <iostream>
#include <sstream>

struct CoverTreeNode {
    CoverTreeNode* parent;
    size_t centroid;
    double coverSize;
    double maxChildDist;
    double maxChildRadius;

    //define node data
    const static bool nodeCmp(std::pair<double,std::unique_ptr<CoverTreeNode>>& a,
                                std::pair<double,std::unique_ptr<CoverTreeNode>>& b) {
        return std::get<0>(a) < std::get<0>(b);
    };
    //stores a distance-sorted list of child nodes
    std::vector<std::pair<double,std::unique_ptr<CoverTreeNode>>> childNodes;

    //define leaf data
    const static bool dataCmp (std::pair<double, size_t>& a,
                                std::pair<double, size_t>& b) {
        return std::get<0>(a) < std::get<0>(b);
    };
    std::vector<std::pair<double,size_t>> childData;
};

template <typename OBJ, typename METRIC>
class CoverTree {

private:
    std::unique_ptr<CoverTreeNode> root;
    //arma::Mat<T> data;

    std::vector<OBJ>& data;
    std::vector<CoverTreeNode*> dataNodes;
    size_t maxLeafSize;
    const double scaleFactor = 1.0/1.3;
    const double minPointSep = 1.0/1.3/1.3/1.3/1.3;

    void addNeighbour(std::vector<std::pair<double,size_t>>& nHeap, const double& dist, const size_t& ind) const {
        /*
        Helper function to add neighbours to the neighbour stack.
        This makes the query code cleaner
        */
        const static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                            const std::pair<double,size_t>& b) {
                                            return std::get<0>(a) < std::get<0>(b);
                                        });

        if (dist < std::get<0>(nHeap.front())) {
            std::pop_heap(nHeap.begin(),
                          nHeap.end(),
                          neighbourCmp);
            nHeap.back() = std::make_pair(dist,ind);
            std::push_heap(nHeap.begin(),
                           nHeap.end(),
                           neighbourCmp);
        }
    }

    void splitNode(CoverTreeNode* c) {
        //TODO: set maxRadius
        if (c->childData.size() > maxLeafSize and c->maxChildDist > c->coverSize*minPointSep) {

            const double epsilon = c->coverSize * scaleFactor;
            const double epsilonHalf = epsilon * 0.5;

            //init the list of children
            std::vector<std::pair<double,size_t>> candidates(c->childData.size(), {std::numeric_limits<double>::max(), size_t(0)});

            //init the first child
            std::get<0>(candidates[0]) = 0.0;
            c->childNodes.push_back({std::get<0>(c->childData[0]), std::unique_ptr<CoverTreeNode>(
                                                    new CoverTreeNode{c,
                                                        size_t(std::get<1>(c->childData[0])),
                                                        epsilon,
                                                        0.0,
                                                        data[std::get<1>(c->childData[0])].radius(),
                                                        std::vector<std::pair<double,std::unique_ptr<CoverTreeNode>>>(),
                                                        std::vector<std::pair<double,size_t>>()}) });

            //NOTE: this is the slowest but foolproof way to build nodes: go through everything and keep assigning until no new node candidates are left
            double maxMeasuredDist = 2*epsilon;
            while (maxMeasuredDist > epsilon) {
                maxMeasuredDist = 0.0;
                size_t nextChild = 0;
                size_t currentChild = std::get<1>(c->childNodes.back())->centroid;
                for (size_t i = 1;
                    i < c->childData.size();
                    ++i) {
                    if (std::get<0>(candidates[i]) > epsilonHalf) {
                        double dist = METRIC::dist(data[std::get<1>(c->childData[i])], data[currentChild]);
                        if (dist < epsilon and dist < std::get<0>(candidates[i])) {
                            candidates[i] = {dist, c->childNodes.size()-1};
                        } else if (dist >= epsilon and nextChild == 0) {
                            nextChild = i;
                        }
                        maxMeasuredDist = std::max(maxMeasuredDist,dist);
                        if (c->childData[i].first - c->childNodes.back().first > epsilon) {
                            maxMeasuredDist = 2*epsilon;
                            break;
                        }
                    }
                }
                if (nextChild != 0) {
                    c->childNodes.push_back({std::get<0>(c->childData[nextChild]), std::unique_ptr<CoverTreeNode>(
                                                    new CoverTreeNode{c,
                                                        size_t(std::get<1>(c->childData[nextChild])),
                                                        epsilon,
                                                        0.0,
                                                        data[std::get<1>(c->childData[nextChild])].radius(),
                                                        std::vector<std::pair<double,std::unique_ptr<CoverTreeNode>>>(),
                                                        std::vector<std::pair<double,size_t>>()}) });
                    dataNodes[c->childData[nextChild].second] = c->childNodes.back().second.get();
                }
            }

            //add children to all the appropriate nodes
            for (size_t i = 1; i < c->childData.size(); ++i) {
                const auto& cand = std::get<1>(c->childNodes[std::get<1>(candidates[i])]);

                if (std::get<1>(c->childData[i]) != cand->centroid) {
                    cand->childData.push_back({std::get<0>(candidates[i]), std::get<1>(c->childData[i])});
                }
                cand->maxChildRadius = std::max(cand->maxChildRadius, data[std::get<1>(c->childData[i])].radius());
            }

            //sort children and split child nodes
            //TODO: this can be parallel
            for (size_t i = 0; i < c->childNodes.size(); ++i) {

                if (std::get<1>(c->childNodes[i])->childData.size() > 1) {
                    const static auto dataCmp = ([](const std::pair<double,size_t>& a,
                                            const std::pair<double,size_t>& b) {
                                            return std::get<0>(a) < std::get<0>(b);
                                        });
                    std::sort(std::get<1>(c->childNodes[i])->childData.begin(), std::get<1>(c->childNodes[i])->childData.end(), dataCmp);
                }

                //set node metadata
                if (c->childNodes[i].second->childData.size() > 0) {
                    c->childNodes[i].second->maxChildDist = c->childNodes[i].second->childData.back().first;
                } else {
                    c->childNodes[i].second->maxChildDist = 0.0;
                }

                //split the completed node
                splitNode(std::get<1>(c->childNodes[i]).get());
            }

            //this forces a reallocation of the child vector to 0
            std::vector<std::pair<double,size_t>>().swap(c->childData);
        }
    }

    void knnQuery_( CoverTreeNode* c,
            std::vector<std::pair<double,size_t>>& neighbourHeap,
            const OBJ&  val,
            double valDist,
            const size_t& kneighbours,
            const size_t& leafLimit,
            CoverTreeNode*& insertNode,
            size_t& leafCount) const {
        //TODO: make this non-recursive

        const static auto indCmp = ([](const std::pair<double,CoverTreeNode*>& a,
                                        const std::pair<double,CoverTreeNode*>& b) {
                                            return std::get<0>(a) > std::get<0>(b);
                                    });

        std::vector<std::pair<double,CoverTreeNode*>> candidates;
        candidates.reserve(8192);
        candidates.push_back({METRIC::dist(data[c->centroid],val),c});
        while (candidates.size() > 0) {
            std::tie(valDist, c) = candidates.front();
            std::pop_heap(candidates.begin(),candidates.end(),indCmp);
            candidates.resize(candidates.size()-1);

            //if there are no children, do the leaf node codepath
            if (c->childNodes.size() == 0) {
                //when querying children, we eliminate as many possibilities as we can before calculating the distance
                auto candidate = c->childData.begin();
                const double vmin = valDist - neighbourHeap.front().first;
                while (candidate != c->childData.end() and candidate->first < vmin) {
                    ++candidate;
                }
                //while candidates are possibly closer than the furthers neighbour, check them and add them to the neighbour heap
                for (; candidate != c->childData.end() and candidate->first < valDist + neighbourHeap.front().first; ++candidate) {
                    const double dist = METRIC::dist(data[candidate->second],val);
                    addNeighbour(neighbourHeap,
                                 dist,
                                 candidate->second);

                }
                if (insertNode == NULL) {
                    insertNode = c;
                }
                ++leafCount;
                if (leafCount >= leafLimit) {
                    break;
                }
            } else {

                //this first step pre-filters candidates by distance for expanding the tree, and adds their centroids to the neighbour heap
                auto candidate = c->childNodes.begin();
                const double vmin = valDist - neighbourHeap.front().first - c->childNodes[0].second->coverSize;
                while (candidate != c->childNodes.end() and candidate->first < vmin) {
                    ++candidate;
                }
                //std::pair<double,CoverTreeNode*> best = {std::numeric_limits<double>::max(), NULL};
                std::vector<std::pair<double,CoverTreeNode*>> round1Cands;
                round1Cands.reserve(c->childNodes.size());
                //for all candidates that could be closer than the furthest neighbour, add them to the candidate heap (and check if they are a closer neighbour)
                for (;candidate != c->childNodes.end() and
                     candidate->first - candidate->second->coverSize < valDist + neighbourHeap.front().first;
                     ++candidate) {
                    if (candidate->first - candidate->second->maxChildDist < valDist + neighbourHeap.front().first) {
                        const double dist = METRIC::dist(data[candidate->second->centroid],val);
                        addNeighbour(neighbourHeap,
                                     dist,
                                     candidate->second->centroid);
                        round1Cands.push_back({dist, candidate->second.get()});
                    }
                }

                //remove candidates which are not candidates any more.
                while (candidates.size() > 0 and candidates.front().first - candidates.front().second->maxChildDist >
                        neighbourHeap.front().first) {
                    std::pop_heap(candidates.begin(),candidates.end(),indCmp);
                    candidates.resize(candidates.size()-1);
                }

                //add new candidates
                for (auto& cand : round1Cands) {
                    if (cand.first - cand.second->maxChildDist <= neighbourHeap.front().first) {
                        candidates.push_back(cand);
                        std::push_heap(candidates.begin(), candidates.end(), indCmp);
                    }
                }
            }
        }
    }

    void ennQuery_(const CoverTreeNode* c,
                std::vector<std::pair<double,size_t>>& neighbours,
                const OBJ&  val,
                const double& valDist,
                const double& eps) const {

        //if there are no children, do the leaf node codepath
        if (c->childNodes.size() == 0) {
            //when querying children, we eliminate as many possibilities as we can before calculating the distance
            const static auto indCmp3 = ([](const std::pair<double, size_t>& a,
                                           const std::pair<double, size_t>& b) {
                                            return std::get<0>(a) < std::get<0>(b);
                                        });
            const double minDist = valDist - eps;
            const double maxDist = valDist + eps;
            //NOTE: this tight loop is faster than a binary search on tested data
            auto candidate = c->childData.begin();
            while (candidate != c->childData.end() and candidate->first < minDist) {
                ++candidate;
            }
            for (;
                 candidate != c->childData.end() and candidate->first < maxDist;
                 ++candidate) {
                const double dist = METRIC::dist(data[candidate->second],val);
                if (dist <= eps) {
                    neighbours.push_back({dist,candidate->second});
                }
            }
            return;
        }
        const double minDist = valDist - eps - c->maxChildDist;
        const double maxDist = valDist + eps + c->maxChildDist;

        //this first step pre-filters candidates by distance for expanding the tree, and adds their centroids to the neighbour heap
        auto candidate = c->childNodes.begin();
        while (candidate != c->childNodes.end() and candidate->first < minDist) {
            ++candidate;
        }
        for (;(candidate != c->childNodes.end()) and (candidate->first < maxDist);
             ++candidate) {
            const double dist = METRIC::dist(data[candidate->second->centroid],val);

            if (dist - candidate->second->maxChildDist <= eps) {
                if (dist <= eps) {
                    neighbours.push_back({dist,candidate->second->centroid});
                }
                ennQuery_(candidate->second.get(),
                       neighbours,
                       val,
                       dist,
                       eps);
            }
        }
    }

    void rennQuery_(const CoverTreeNode* c,
                std::vector<std::pair<double,size_t>>& neighbours,
                const OBJ&  val,
                const double& valDist) const {

        //if there are no children, do the leaf node codepath
        if (c->childNodes.size() == 0) {
            //when querying children, we eliminate as many possibilities as we can before calculating the distance
            const static auto indCmp3 = ([](const std::pair<double, size_t>& a,
                                           const std::pair<double, size_t>& b) {
                                            return std::get<0>(a) < std::get<0>(b);
                                        });
            //NOTE: this tight loop is faster than a binary search on tested data
            auto candidate = c->childData.begin();
            while (candidate != c->childData.end() and candidate->first < valDist - c->maxChildRadius) {
                ++candidate;
            }
            for (;
                 candidate != c->childData.end() and candidate->first < valDist + c->maxChildRadius;
                 ++candidate) {
                const double dist = METRIC::dist(data[candidate->second],val);
                if (dist <=  data[candidate->second].radius()) {
                    neighbours.push_back({dist,candidate->second});
                }
            }
            return;
        }
        const double minDist = valDist - c->maxChildDist - c->maxChildRadius;
        const double maxDist = valDist + c->maxChildDist + c->maxChildRadius;

        //this first step pre-filters candidates by distance for expanding the tree, and adds their centroids to the neighbour heap
        auto candidate = c->childNodes.begin();
        while (candidate != c->childNodes.end() and candidate->first < minDist) {
            ++candidate;
        }
        for (;(candidate != c->childNodes.end()) and (candidate->first < maxDist);
             ++candidate) {
            const double dist = METRIC::dist(data[candidate->second->centroid],val);

            if (dist - candidate->second->maxChildDist <= candidate->second->maxChildRadius) {
                if (dist <=  data[candidate->second->centroid].radius()) {
                    neighbours.push_back({dist,candidate->second->centroid});
                }
                rennQuery_(candidate->second.get(),
                       neighbours,
                       val,
                       dist);
            }
        }
    }

public:
     typedef CoverTreeNode* nodeReturnType;

    CoverTree(std::vector<OBJ>& d, const size_t leafSize = 100) : data(d) {
        dataNodes.resize(data.size());
        //first create a distance-to-root-sorted list of all data indices (root is always index 0)
        std::vector<std::pair<double,size_t>> inds;
        inds.reserve(data.size()-1);
        double maxRadius = data[0].radius();
        for (size_t i = 1; i < data.size(); ++i) {
            inds.push_back({METRIC::dist(data[0], data[i]), i});
            maxRadius = std::max(maxRadius, data[i].radius());
        }

        //save the largest size so we can set it to be the root cover radius
        double epsilon = 0.0;
        if (inds.size() > 1) {
            const static auto dataCmp = ([](const std::pair<double,size_t>& a,
                                            const std::pair<double,size_t>& b) {
                                            return std::get<0>(a) < std::get<0>(b);
                                        });
            std::sort(inds.begin(),inds.end(),dataCmp);
            epsilon = std::get<0>(inds.back());
        }

        maxLeafSize = leafSize;

        //make the root node
        root = std::unique_ptr<CoverTreeNode>(
                                     new CoverTreeNode{NULL,
                                            size_t(0),
                                            epsilon,
                                            epsilon,
                                            maxRadius,
                                            std::vector<std::pair<double,std::unique_ptr<CoverTreeNode>>>(),
                                            inds}
                                           );
        dataNodes[0] = root.get();

        //split the root node
        splitNode(root.get());

    }

    ///Update the tree max-radius values for the item at dataIndex, which used to have the value oldRadius
    void updateRadius(const size_t& dataIndex, const double& oldRadius) {
        //let the tree know a data item radius has been updated, and propagate the changes through the tree.
        auto currentNode = dataNodes[dataIndex];
        if (data[dataIndex].radius() > currentNode->maxChildRadius) { //this is the case where we are a new maxradius
            while (currentNode != NULL and currentNode->maxChildRadius < data[dataIndex].radius()) {
                currentNode->maxChildRadius = data[dataIndex].radius();
                currentNode = currentNode->parent;
            }
        } else {
            while (currentNode != NULL and currentNode->maxChildRadius == oldRadius) { //this is the case where we are an old maxradius
                if (currentNode->childData.size() > 0) {
                    currentNode->maxChildRadius = data[currentNode->centroid].radius();
                    for (const auto& c: currentNode->childData) {
                        currentNode->maxChildRadius = std::max(
                                            currentNode->maxChildRadius,
                                            data[c.second].radius());
                    }
                } else {
                    currentNode->maxChildRadius = data[currentNode->centroid].radius();
                    for (const auto& c: currentNode->childNodes) {
                        currentNode->maxChildRadius = std::max(
                                            currentNode->maxChildRadius,
                                            data[c.second->centroid].radius());
                    }
                }
                currentNode = currentNode->parent;
            }
        }
    }

    std::vector<std::pair<double,size_t>> knnQuery(const OBJ& val,
                                                    const size_t& kneighbours,
                                                    CoverTreeNode*& insertNode,
                                                    const size_t& maxLeaves = std::numeric_limits<size_t>::max()) const {
        /*
        Depth-based query for kneighbours closest points to val,
        using a maximum of maxLeaves number of leaf searches.
        NOTE: maxLeaves should be 1 for spill-tree search,
        since double values are not filtered.
        */

        //initialise heaps

        std::vector<std::pair<double,size_t>> neighbourHeap(kneighbours,
                                                    {std::numeric_limits<double>::max(), size_t(0)});
        const double& dist = METRIC::dist(data[root->centroid],val);
        addNeighbour(neighbourHeap, dist, root->centroid);
        insertNode = NULL;
        size_t leaves = 0;
        knnQuery_(root.get(),
                neighbourHeap,
                val,
                dist,
                kneighbours,
                maxLeaves,
                insertNode,
                leaves);

        return neighbourHeap;
    }

    std::vector<std::pair<double,size_t>> knnQuery(const OBJ& val,
                                                    const size_t& kneighbours) const {
        /*
        Depth-based query for kneighbours closest points to val,
        using a maximum of maxLeaves number of leaf searches.
        NOTE: maxLeaves should be 1 for spill-tree search,
        since double values are not filtered.
        */

        //initialise heaps

        CoverTreeNode* c;

        return knnQuery(val,kneighbours,c,std::numeric_limits<size_t>::max());
    }

    std::vector<std::pair<double,size_t>> ennQuery(const OBJ& val,
                                                   const double& eps) const {
        /*
        Depth-based query for kneighbours closest points to val,
        using a maximum of maxLeaves number of leaf searches.
        NOTE: maxLeaves should be 1 for spill-tree search,
        since double values are not filtered.
        */

        //initialise heaps
        std::vector<std::pair<double,size_t>> neighbours;
        neighbours.reserve(1000);

        const double& dist = METRIC::dist(data[root->centroid],val);
        if (dist <= eps) {
            neighbours.push_back({dist,root->centroid});
        }

        ennQuery_(root.get(),
                neighbours,
                val,
                dist,
                eps);

        return neighbours;
    }

    std::vector<std::pair<double,size_t>> rennQuery(const OBJ& val) const {
        /*
        Depth-based query for kneighbours closest points to val,
        using a maximum of maxLeaves number of leaf searches.
        NOTE: maxLeaves should be 1 for spill-tree search,
        since double values are not filtered.
        */

        //initialise heaps
        std::vector<std::pair<double,size_t>> neighbours;
        neighbours.reserve(1000);

        const double& dist = METRIC::dist(data[root->centroid],val);
        if (dist <= data[root->centroid].radius()) {
            neighbours.push_back({dist,root->centroid});
        }
        rennQuery_(root.get(),
                neighbours,
                val,
                dist);

        return neighbours;
    }

    void insert(OBJ& item, CoverTreeNode* node = NULL) {
        //set target node if a valid node is not provided
        double dist;
        if (node == NULL) {
            CoverTreeNode* nextNode = root.get();
            dist = METRIC::dist(data[nextNode->centroid],item);
            if (dist < nextNode->coverSize) {
                nextNode->maxChildDist = std::max(dist,nextNode->maxChildDist);
                nextNode->maxChildRadius = std::max(item.radius(),nextNode->maxChildRadius);
            }

            while ( (node == NULL or node->childData.size() == 0) and nextNode != node) {
                node = nextNode;
                if (node->childNodes.size() > 0) {
                    double minDist = dist - node->childNodes[0].second->coverSize;
                    double maxDist = dist + node->childNodes[0].second->coverSize;
                    dist = std::numeric_limits<double>::max();
                    size_t i = 0;
                    //we can skip evaluating some candidates, since we know in advance they can't cover the item
                    while (i < node->childNodes.size() and
                           node->childNodes[i].first < minDist ) {
                        ++i;
                    }
                    for (; i < node->childNodes.size() and node->childNodes[i].first < maxDist; ++i) {
                        double cdist = METRIC::dist(data[node->childNodes[i].second->centroid],item);
                        if (cdist < dist and cdist < node->childNodes[i].second->coverSize) {
                            dist = cdist;
                            nextNode = node->childNodes[i].second.get();

                        }
                    }
                    if (dist < std::numeric_limits<double>::max()) {
                        nextNode->maxChildDist = std::max(dist,nextNode->maxChildDist);
                        nextNode->maxChildRadius = std::max(item.radius(),nextNode->maxChildRadius);
                    }
                }
            }
        } else {
            //CoverTreeNode* nextNode = root.get();
            while (node != NULL) {
                dist = METRIC::dist(data[node->centroid],item);
                if (dist < node->coverSize) {
                    break;
                }
                node = node->parent;
            }

            //update maxchilddist
            for (auto p = node; p != NULL; p = p->parent) {
                dist = METRIC::dist(data[p->centroid],item);
                p->maxChildDist = std::max(dist,p->maxChildDist);
                p->maxChildRadius = std::max(item.radius(),p->maxChildRadius);
            }

            if (node == NULL) {
                node = root.get();
            }

        }


        dist = METRIC::dist(data[node->centroid],item);
        if (node == root.get() and dist > node->coverSize) {

            double newCoverFactor = root->coverSize / scaleFactor;
            if (node->childNodes.size() == 0) {
                //insert item
                node->coverSize = dist;
                node->maxChildDist = dist;
                node->childData.push_back({dist,data.size()});
                data.push_back(item);
                dataNodes.push_back(node);
                if (node->childData.size() > 1) {
                    const static auto dataCmp2 = ([](const std::pair<double,size_t>& a,
                                                    const std::pair<double,size_t>& b) {
                                                    return std::get<0>(a) < std::get<0>(b);
                                                });
                    std::sort(node->childData.begin(), node->childData.end(), dataCmp2);
                }
                splitNode(node);
            } else if (dist <= newCoverFactor + root->maxChildDist) {

                //create a new root
                auto tmp = std::unique_ptr<CoverTreeNode>(
                                                        new CoverTreeNode{NULL,
                                                            size_t(data.size()),
                                                            newCoverFactor,
                                                            std::min(root->maxChildDist + dist, newCoverFactor),
                                                            std::max(item.radius(), root->maxChildRadius),
                                                            std::vector<std::pair<double,std::unique_ptr<CoverTreeNode>>>(),
                                                            std::vector<std::pair<double,size_t>>()});
                root->parent = tmp.get();
                std::swap(root,tmp);
                //dataNodes[tmp->centroid] = tmp.get();
                root->childNodes.push_back({dist, std::move(tmp)});
                data.push_back(item);
                dataNodes.push_back(root.get());
            } else {
                std::cout << "ERROR: RESIZING MULTIPLE LEVELS IS NOT IMPLEMENTED YET" << std::endl;
                //TODO: implement the multi-level resize.
                // This is done by sequentially making data items in leaf nodes into root nodes until the new item is covered.
            }

        } else if (node->childData.size() == 0 and node->childNodes.size() != 0) {

            //create new node
            node->childNodes.push_back({dist, std::unique_ptr<CoverTreeNode>(
                                                    new CoverTreeNode{node,
                                                        size_t(data.size()),
                                                        node->coverSize * scaleFactor,
                                                        0.0,
                                                        item.radius(),
                                                        std::vector<std::pair<double,std::unique_ptr<CoverTreeNode>>>(),
                                                        std::vector<std::pair<double,size_t>>()}) });
            data.push_back(item);
            dataNodes.push_back(node->childNodes.back().second.get());
            if (node->childNodes.size() > 1) {
                const static auto dataCmp2 = ([](const std::pair<double,std::unique_ptr<CoverTreeNode>>& a,
                                                const std::pair<double,std::unique_ptr<CoverTreeNode>>& b) {
                                                return std::get<0>(a) < std::get<0>(b);
                                            });
                std::sort(node->childNodes.begin(), node->childNodes.end(), dataCmp2);
            }

        } else {

            //insert item
            node->childData.push_back({dist,data.size()});
            data.push_back(item);
            dataNodes.push_back(node);
            if (node->childData.size() > 1) {
                const static auto dataCmp2 = ([](const std::pair<double,size_t>& a,
                                                const std::pair<double,size_t>& b) {
                                                return std::get<0>(a) < std::get<0>(b);
                                            });
                std::sort(node->childData.begin(), node->childData.end(), dataCmp2);
            }
            splitNode(node);
        }

        //update the tree constraints
        updateRadius(data.size()-1, 0.0);
    }
};

#endif

