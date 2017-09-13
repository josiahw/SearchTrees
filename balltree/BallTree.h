
#ifndef BALLTREE_H
#define BALLTREE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <sys/types.h>


struct BallTreeNode {
    size_t parentIndex;

    //indicates the dimension to split
    size_t splitPoint;

    //indicates the value for the split plane
    double splitVal;

    //indicates the maximum radius of the child objects - reverse-KNN only
    double maxRadius;

    //stores the indices for child data objects (only if this node is a leaf)
    std::vector<size_t> childData;
    std::vector<size_t> spillChildData;

    //stores the ID's for child vp-tree nodes (only if this node is not a leaf)
    size_t leftChildNode;
    size_t rightChildNode;
};

//The BallTree is a VP-Tree implementation which works in general metric spaces
template <typename OBJ, typename METRIC>
class BallTree {

private:
    size_t root;
    size_t maxLeafSize;
    double spillEps;
    std::vector<BallTreeNode> nodes;
    std::vector<OBJ>& data;
    std::vector<size_t> dataNodes;

    void addNeighbour(std::vector< std::pair<double, size_t> >& nHeap,
                      const double& dist,
                      const size_t& ind) const {
        static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
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
    };

    ///split an existing node to create new leaf nodes
    void splitNode(const size_t& parentIndex) {
        //NOTE: THIS RESERVE MAY RELOCATE THE CONST AUTO& VALUES, SO MUST GO FIRST
        nodes.reserve(nodes.size()+2);

        const auto& dataIndices = nodes[parentIndex].childData;
        if (dataIndices.size() <= maxLeafSize) {
            return;
        }

        size_t centrePoint = 0;
        double splitVal = 0.;
        double spread = -1.;
        arma::Col<double> vals(dataIndices.size());
        arma::Col<double> dists(dataIndices.size());
        //pick the candidate with the best spread from 20 candidates.
        arma::uvec cands = arma::randi<arma::uvec>(20, arma::distr_param(0, dataIndices.size()-1));
        for (size_t i = 0; i < cands.n_elem; ++i) {

            #pragma omp parallel for
            for (size_t j = 0; j < dataIndices.size(); ++j) {
                dists[j] = METRIC::dist(data[dataIndices[j]], data[dataIndices[cands[i]]]);
            }
            const double spltmp = arma::median(arma::Col<double>(dists));
            const double sprtmp = arma::accu(arma::abs(dists-spltmp));
            if (sprtmp > spread) {
                splitVal = spltmp;
                centrePoint = dataIndices[cands[i]];
                spread = sprtmp;
                arma::swap(vals,dists);
            }
        }
        size_t splitPt = centrePoint;

        if (spread <= 0.000000001) {
            return;
        }

        nodes[parentIndex].splitVal = splitVal;
        nodes[parentIndex].splitPoint = splitPt;

        //separate the left and right child nodes
        std::vector<size_t> leftChildren, rightChildren, leftSpill, rightSpill;
        leftChildren.reserve(dataIndices.size()/2+1);
        rightChildren.reserve(dataIndices.size()/2+1);
        for (size_t i = 0; i < dataIndices.size(); ++i) {
            if (dataIndices[i] != splitPt) {
                if (vals[i] <= splitVal) {
                    leftChildren.push_back(dataIndices[i]);
                    if (vals[i] + spillEps > splitVal) {
                        rightSpill.push_back(dataIndices[i]);
                    }
                } else {
                    rightChildren.push_back(dataIndices[i]);
                    if (vals[i] - spillEps < splitVal) {
                        leftSpill.push_back(dataIndices[i]);
                    }
                }
            }
        }


        //push back inherited spill indices for each child
        for (const auto& c : nodes[parentIndex].spillChildData) {
            const double d = METRIC::dist(data[c], data[splitPt]);

            if (d + spillEps > splitVal) {
                rightSpill.push_back(c);
            }
            if (d - spillEps < splitVal) {
                leftSpill.push_back(c);
            }
        }


        std::vector<size_t>().swap(nodes[parentIndex].childData);
        std::vector<size_t>().swap(nodes[parentIndex].spillChildData);

        nodes[parentIndex].leftChildNode = mkNodeRecursive(parentIndex,
                                                        leftChildren,
                                                        leftSpill);
        nodes[parentIndex].rightChildNode = mkNodeRecursive(parentIndex,
                                                        rightChildren,
                                                        rightSpill);

    }

    size_t mkNodeRecursive(const size_t& parentIndex,
                           const std::vector<size_t>& dataIndices,
                           const std::vector<size_t>& spillIndices) {

        //early exit to save processing if the leaf size is too small
        if (dataIndices.size() <= maxLeafSize ) {
            //return a leaf node
            const size_t empty = 0;
            double maxRadius = 0.;
            nodes.push_back(BallTreeNode({parentIndex,
                                        0,
                                        0.,
                                        maxRadius,
                                        dataIndices,
                                        spillIndices,
                                        empty,
                                        empty}));
            for (size_t i = 0; i < dataIndices.size(); ++i) {
              maxRadius = std::max(data[dataIndices[i]].radius(), maxRadius);
              dataNodes[dataIndices[i]] = nodes.size()-1;
            }
            return nodes.size()-1;
        }


        size_t centrePoint = 0;
        double splitVal = 0.;
        double spread = -1.;
        arma::Col<double> vals(dataIndices.size());
        arma::Col<double> dists(dataIndices.size());
        //pick the candidate with the best spread from 20 candidates.
        arma::uvec cands = arma::randi<arma::uvec>(20, arma::distr_param(0, dataIndices.size()-1));
        for (size_t i = 0; i < cands.n_elem; ++i) {
            for (size_t j = 0; j < dataIndices.size(); ++j) {
                dists[j] = METRIC::dist(data[dataIndices[j]], data[dataIndices[cands[i]]]);
            }
            const double spltmp = arma::median(arma::Col<double>(dists));
            const double sprtmp = arma::accu(arma::abs(dists-spltmp));
            if (sprtmp > spread) {
                splitVal = spltmp;
                centrePoint = dataIndices[cands[i]];
                spread = sprtmp;
                arma::swap(vals,dists);
            }
        }
        dists.reset();
        cands.reset();
        size_t splitPt = centrePoint;

        if (spread <= 0.000000001) {
            //return a leaf node
            const size_t empty = 0;
            double maxRadius = 0.;
            for (size_t i = 0; i < dataIndices.size(); ++i) {
              maxRadius = std::max(data[dataIndices[i]].radius(), maxRadius);
              dataNodes[dataIndices[i]] = nodes.size()-1;
            }
            nodes.push_back(BallTreeNode({parentIndex,
                                        0,
                                        0.,
                                        maxRadius,
                                        dataIndices,
                                        spillIndices,
                                        empty,
                                        empty}));
            for (size_t i = 0; i < dataIndices.size(); ++i) {
              maxRadius = std::max(data[dataIndices[i]].radius(), maxRadius);
              dataNodes[dataIndices[i]] = nodes.size()-1;
            }
            return nodes.size()-1;
        }

        //separate the left and right child nodes
        std::vector<size_t> leftChildren, rightChildren, leftSpill, rightSpill;
        leftChildren.reserve(dataIndices.size()/2+1);
        rightChildren.reserve(dataIndices.size()/2+1);
        double maxRadius = 0.;
        for (size_t i = 0; i < dataIndices.size(); ++i) {
            maxRadius = std::max(data[dataIndices[i]].radius(), maxRadius);
            if (dataIndices[i] != splitPt) {
                if (vals[i] <= splitVal) {
                    leftChildren.push_back(dataIndices[i]);
                    if (vals[i] + spillEps > splitVal) {
                        rightSpill.push_back(dataIndices[i]);
                    }
                } else {
                    rightChildren.push_back(dataIndices[i]);
                    if (vals[i] - spillEps < splitVal) {
                        leftSpill.push_back(dataIndices[i]);
                    }
                }
            }
        }
        vals.reset();
        //push back inherited spill indices for each child
        for (const auto& c : spillIndices) {
            const double d = METRIC::dist(data[c], data[splitPt]);

            if (d + spillEps > splitVal) {
                rightSpill.push_back(c);
            }
            if (d - spillEps < splitVal) {
                leftSpill.push_back(c);
            }
        }

        size_t currentIndex = nodes.size();
        //make a new node with children
        nodes.push_back(BallTreeNode({
                parentIndex,
                splitPt,
                splitVal,
                maxRadius,
                std::vector<size_t>(),
                std::vector<size_t>(),
                mkNodeRecursive(currentIndex,
                                leftChildren,
                                leftSpill),
                mkNodeRecursive(currentIndex,
                                rightChildren,
                                rightSpill)
              }));
        return nodes.size()-1;
    };

public:
    typedef size_t nodeReturnType;

    ///create a BallTree with a dataset d, and a default maximum leaf-node size of 100 children
    BallTree(std::vector<OBJ>& d,
            const double& spillEps = 0.0,
            const size_t& mLeafSize = 100) : data(d), spillEps(spillEps) {
        maxLeafSize = mLeafSize;
        dataNodes.resize(data.size());
        dataNodes.reserve(data.capacity());
        dataNodes[0] = 0;
        nodes.reserve(data.capacity());
        std::vector<size_t> dataIndices;
        dataIndices.resize(data.size());
        for (size_t i = 0; i < dataIndices.size(); ++i) {
            dataIndices[i] = i;
        }
        root = mkNodeRecursive(0,
                               dataIndices,
                               std::vector<size_t>());
    }

    ///Update the tree max-radius values for the item at dataIndex, which used to have the value oldRadius
    void updateRadius(const size_t& dataIndex, const double& oldRadius) {
        //let the tree know a data item radius has been updated, and propagate the changes through the tree.
        //std::cout << "Index: " << dataIndex << std::endl;
        size_t currentNode = dataNodes[dataIndex];
        //std::cout << "CurrentNode: " << currentNode << std::endl;
        if (data[dataIndex].radius() > nodes[currentNode].maxRadius) { //this is the case where we are a new maxradius
            //std::cout << "doing maxradius replacement" << std::endl;
            while (nodes[currentNode].maxRadius < data[dataIndex].radius()) {
                nodes[currentNode].maxRadius = data[dataIndex].radius();
                currentNode = nodes[currentNode].parentIndex;
                if (nodes[currentNode].parentIndex == currentNode) {
                    break;
                }
            }
        } else {
            while (nodes[currentNode].maxRadius == oldRadius) { //this is the case where we are an old maxradius
                //std::cout << "maxradius has changed" << std::endl;
                if (nodes[currentNode].childData.size() > 0) {
                    nodes[currentNode].maxRadius = 0.0;
                    //std::cout << "getting max childdata radius" << std::endl;
                    for (const auto& c: nodes[currentNode].childData) {
                        nodes[currentNode].maxRadius = std::max(
                                            nodes[currentNode].maxRadius,
                                            data[c].radius());
                    }
                } else {
                    //std::cout << "getting max childnode radius" << std::endl;
                    nodes[currentNode].maxRadius = std::max(
                                            nodes[nodes[currentNode].leftChildNode].maxRadius,
                                            nodes[nodes[currentNode].rightChildNode].maxRadius);
                }
                if (nodes[currentNode].parentIndex == currentNode) {
                    break;
                }
                currentNode = nodes[currentNode].parentIndex;
            }
        }
    }

    ///do a KNN query for val, with an optional leaf limit
    std::vector<std::pair<double,size_t>> knnQuery(const OBJ& val,
                                                    const size_t& kneighbours,
                                                    size_t& homeNode,
                                                    const size_t& maxLeaves=800000,
                                                    const double& s1 = 1.0,
                                                    const double& s2 = 1.0) const {
        bool homeSet = false;

        //initialise heaps
        std::vector<std::tuple<double,
                               size_t>> candidateHeap(1,std::make_tuple(0.0,root));
        candidateHeap.reserve(4096); //XXX: this is a dirty way of avoiding some memory copies
        std::vector<std::pair<double,size_t>> neighbourHeap(kneighbours,
                                                    {std::numeric_limits<double>::max(), 0});
        //initialise heap comparisons
        auto candidateCmp = ([](const std::tuple<double,size_t>& a,
            const std::tuple<double,size_t>& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

        size_t leafCounter = 0;
        //keep going until the closest KD-tree node is further than the furthest neighbour
        while (candidateHeap.size() > 0
               and std::get<0>(candidateHeap[0]) < std::get<0>(neighbourHeap[0])
               and leafCounter < maxLeaves) {

            //get a new candidate KD-tree node
            std::pop_heap(candidateHeap.begin(),
                          candidateHeap.end(),
                          candidateCmp);
            auto current = candidateHeap.back();
            candidateHeap.resize(candidateHeap.size()-1);


            while (nodes[std::get<1>(current)].childData.size() == 0) {
                const BallTreeNode& cnode = nodes[std::get<1>(current)];
                //descend until we reach a leaf
                double pDist = METRIC::dist(data[cnode.splitPoint], val);
                addNeighbour(neighbourHeap, pDist, cnode.splitPoint);
                size_t farChild;
                if (pDist > cnode.splitVal) {
                    farChild = cnode.leftChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.rightChildNode);
                    pDist = std::max(pDist-cnode.splitVal, std::get<0>(current));
                } else {
                    farChild = cnode.rightChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.leftChildNode);
                    pDist = std::max(cnode.splitVal-pDist, std::get<0>(current));
                }

                //put the more distant decision node onto the heap for later
                candidateHeap.push_back(std::make_pair(pDist, farChild));
                std::push_heap(candidateHeap.begin(),
                               candidateHeap.end(),
                               candidateCmp);
                }

            //add any near neighbours to the result
            const BallTreeNode& cnode = nodes[std::get<1>(current)];
            for (int i = 0; i < cnode.childData.size(); ++i) {
                double dist = METRIC::dist(data[cnode.childData[i]], val);
                addNeighbour(neighbourHeap, dist, cnode.childData[i]);
            }
            if (not homeSet) {
                homeNode = std::get<1>(current);
                homeSet = true;
            }
            ++leafCounter;
        }
    return neighbourHeap;
    };

    std::vector<std::pair<double,size_t>> knnQuery(const OBJ& val,
                                                    const size_t& kneighbours) {
        size_t homeNodeRef;
        return knnQuery(val, kneighbours, homeNodeRef);
    }

    ///do a reverse-ENN query for val, with an optional leaf limit
    std::vector<std::pair<double,size_t>> rennQuery(const OBJ& val,
                                                    const size_t& maxLeaves=800000) const {
        //initialise heaps
        std::vector<std::tuple<double,
                               size_t>> candidateHeap(1,std::make_tuple(0.0,root));
        candidateHeap.reserve(4096); //XXX: this is a dirty way of avoiding some memory copies
        std::vector<std::pair<double,size_t>> neighbourStack;
        neighbourStack.reserve(1000);

        //initialise heap comparisons
        auto candidateCmp = ([](const std::tuple<double,size_t>& a,
            const std::tuple<double,size_t>& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

        size_t leafCounter = 0;

        //keep going until the closest KD-tree node is further than the furthest neighbour
        while (candidateHeap.size() > 0
               and leafCounter < maxLeaves) {

            //get a new candidate KD-tree node
            auto current = candidateHeap.back();
            candidateHeap.resize(candidateHeap.size()-1);

            while (nodes[std::get<1>(current)].childData.size() == 0
                    and nodes[std::get<1>(current)].maxRadius >= std::get<0>(current)
                    ) {
                const BallTreeNode& cnode = nodes[std::get<1>(current)];
                //descend until we reach a leaf
                double pDist = METRIC::dist(data[cnode.splitPoint], val);
                if (pDist <= data[cnode.splitPoint].radius()) {
                    neighbourStack.push_back({pDist, cnode.splitPoint});
                }
                size_t farChild;
                if (pDist > cnode.splitVal) {
                    farChild = cnode.leftChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.rightChildNode);
                    pDist = std::max(pDist-cnode.splitVal, std::get<0>(current));
                } else {
                    farChild = cnode.rightChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.leftChildNode);
                    pDist = std::max(cnode.splitVal-pDist, std::get<0>(current));
                }

                //put the more distant decision node onto the heap for later
                if (nodes[farChild].maxRadius >= pDist - cnode.splitVal) {
                    candidateHeap.push_back(std::make_pair(pDist, farChild));
                }
            }
            //add any near neighbours to the result
            const BallTreeNode& cnode = nodes[std::get<1>(current)];
            for (const auto& c : cnode.childData) {
                double dist = METRIC::dist(data[c], val);
                if (dist <= data[c].radius()) {
                    neighbourStack.push_back({dist, c});
                }
            }
            ++leafCounter;
        }
    return neighbourStack;
    };

    ///do a epsilon-NN query for val, with an optional leaf limit
    std::vector<std::pair<double,size_t>> ennQuery(const OBJ& val,
                                                    const double& eps,
                                                    const size_t& maxLeaves=800000) const {
        //initialise heaps
        std::vector<std::tuple<double,
                               size_t>> candidateHeap(1,std::make_tuple(0.0,root));
        candidateHeap.reserve(4096); //XXX: this is a dirty way of avoiding some memory copies
        std::vector<std::pair<double,size_t>> neighbourStack;
        neighbourStack.reserve(1000);

        size_t leafCounter = 0;
        //keep going until the closest KD-tree node is further than the furthest neighbour
        while (candidateHeap.size() > 0
               and leafCounter < maxLeaves) {

            //get a new candidate KD-tree node
            auto current = candidateHeap.back();
            candidateHeap.resize(candidateHeap.size()-1);


            while (nodes[std::get<1>(current)].childData.size() == 0) {
                const BallTreeNode& cnode = nodes[std::get<1>(current)];
                //descend until we reach a leaf
                double pDist = METRIC::dist(data[cnode.splitPoint], val);
                if (pDist < eps) {
                    neighbourStack.push_back({pDist, cnode.splitPoint});
                }
                size_t farChild;
                if (pDist > cnode.splitVal) {
                    farChild = cnode.leftChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.rightChildNode);
                    pDist = std::max(pDist-cnode.splitVal, std::get<0>(current));
                } else {
                    farChild = cnode.rightChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.leftChildNode);
                    pDist = std::max(cnode.splitVal-pDist, std::get<0>(current));
                }

                //put the more distant decision node onto the heap for later
                if (pDist < eps) {
                    candidateHeap.push_back(std::make_pair(pDist, farChild));
                }
                }

            //add any near neighbours to the result
            const BallTreeNode& cnode = nodes[std::get<1>(current)];
            for (int i = 0; i < cnode.childData.size(); ++i) {
                double dist = METRIC::dist(data[cnode.childData[i]], val);
                if (dist < eps) {
                    neighbourStack.push_back({dist, cnode.childData[i]});
                }
            }
            ++leafCounter;
        }
    return neighbourStack;
    };

    //insert takes an object and an optional leaf node index and inserts the object into the tree
    void insert(OBJ& item, size_t node = std::numeric_limits<size_t>::max()) {

        //set target node if a valid node is not provided

        //std::cout << "Finding insert node" << std::endl;
        const auto& val = item.value();
        if (node > nodes.size()) {
            node = root;
            while (nodes[node].childData.size() == 0) {
                if (METRIC::dist(data[nodes[node].splitPoint], item) <= nodes[node].splitVal) {
                    node = nodes[node].leftChildNode;
                } else {
                    node = nodes[node].rightChildNode;
                }
            }
        }

        //std::cout << "inserting item" << std::endl;
        //insert item
        nodes[node].childData.push_back(data.size());
        data.push_back(item);
        dataNodes.push_back(node);

        //update the tree constraints
        //std::cout << "updating radius for insert" << std::endl;
        updateRadius(data.size()-1, 0.0);

        //std::cout << "splitting node" << std::endl;
        splitNode(node);

        //std::cout << "done" << std::endl;
    }

    //TODO: maybe delete and knn-renn query? Optional at this stage.
    //TODO2: fix distance estimation to be more awesome.
};

#endif

