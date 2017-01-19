
#ifndef KDTREE_H
#define KDTREE_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <limits>
#include <sys/types.h>


struct KDTreeNode {
    size_t parentIndex;

    //indicates the dimension to split
    size_t splitDim;

    //indicates the value for the split plane
    double splitVal;

    //indicates the maximum radius of the child objects - reverse-KNN only
    double maxRadius;

    //indicates the bounds for child data (maybe we should augment this to be a BiH-tree?)
    arma::Col<double> LowerBounds;
    arma::Col<double> UpperBounds;

    //stores the indices for child data objects (only if this node is a leaf)
    std::vector<size_t> childData;
    std::vector<size_t> spillChildData;

    //stores the ID's for child kd-tree nodes (only if this node is not a leaf)
    size_t leftChildNode;
    size_t rightChildNode;
};


template <typename OBJ, typename METRIC>
class KDTree {

private:
    size_t root;
    size_t maxLeafSize;
    double spillEps;
    std::vector<KDTreeNode> nodes;
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
        //NOTE: THIS MAY RELOCATE THE CONST AUTO& VALUES, SO MUST GO FIRST
        nodes.reserve(nodes.size()+2);

        const auto& dataIndices = nodes[parentIndex].childData;
        if (dataIndices.size() <= maxLeafSize) {
            return;
        }
        const auto& LowerBounds = nodes[parentIndex].LowerBounds;
        const auto& UpperBounds = nodes[parentIndex].UpperBounds;

        const arma::Col<double> boundGap = arma::conv_to<arma::Col<double>>::from(UpperBounds-LowerBounds);

        size_t splitDim = arma::index_max(boundGap);

        if (boundGap[splitDim]*boundGap[splitDim] <= 0.000000001) {
            return;
        }

        //build a vector along the filtered dimension
        arma::rowvec vals(dataIndices.size());
        for (size_t i = 0; i < dataIndices.size(); ++i) {
          vals[i] = data[dataIndices[i]].value()[splitDim];
        }

        //use the median for the split plane value
        double splitVal = arma::median( vals );
        //check if the median actually divides the points, if not use the midpoint
        double maxv = arma::max(vals);
        if (maxv == splitVal) {
            double minv = arma::min(vals);
            splitVal = (maxv+minv)/2.;
        }

        nodes[parentIndex].splitVal = splitVal;
        nodes[parentIndex].splitDim = splitDim;

        //create the update child bounds
        arma::Col<double> leftBounds = UpperBounds;
        leftBounds[splitDim] = splitVal;
        arma::Col<double> rightBounds = LowerBounds;
        rightBounds[splitDim] = splitVal;

        //separate the left and right child nodes
        std::vector<size_t> leftChildren, rightChildren, leftSpill, rightSpill;
        leftChildren.reserve(dataIndices.size()/2+1);
        rightChildren.reserve(dataIndices.size()/2+1);
        for (size_t i = 0; i < vals.n_elem; ++i) {
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

        //push back spill data from parent spill data
        for (const auto& c : nodes[parentIndex].spillChildData) {
            if (data[c].value()[splitDim] + spillEps > splitVal) {
                rightSpill.push_back(c);
            }
            if (data[c].value()[splitDim] - spillEps < splitVal) {
                leftSpill.push_back(c);
            }
        }


        nodes[parentIndex].childData.resize(0);
        nodes[parentIndex].spillChildData.resize(0);

        nodes[parentIndex].leftChildNode = mkNodeRecursive(parentIndex,
                                                        leftChildren,
                                                        leftSpill,
                                                        LowerBounds,
                                                        leftBounds);
        nodes[parentIndex].rightChildNode = mkNodeRecursive(parentIndex,
                                                        rightChildren,
                                                        rightSpill,
                                                        rightBounds,
                                                        UpperBounds);

    }

    size_t mkNodeRecursive(const size_t& parentIndex,
                           const std::vector<size_t>& dataIndices,
                           const std::vector<size_t>& spillIndices,
                           const arma::Col<double>& LowerBounds,
                           const arma::Col<double>& UpperBounds) {
        const arma::Col<double> boundGap = UpperBounds-LowerBounds;
        size_t splitDim = arma::index_max(boundGap);

        //early exit to save processing if the leaf size is too small / points can't be separated
        if (dataIndices.size() <= maxLeafSize or boundGap[splitDim]*boundGap[splitDim] <= 0.000000001) {
            //return a leaf node
            const size_t empty = 0;
            double maxRadius = 0.;

            nodes.push_back(KDTreeNode({parentIndex,
                                        0,
                                        0.,
                                        maxRadius,
                                        LowerBounds,
                                        UpperBounds,
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

        //build a vector along the filtered dimension
        arma::rowvec vals(dataIndices.size());
        double maxRadius = 0.;
        for (size_t i = 0; i < dataIndices.size(); ++i) {
          vals[i] = data[dataIndices[i]].value()[splitDim];
          maxRadius = std::max(data[dataIndices[i]].radius(), maxRadius);
        }

        //use the median for the split plane value
        double splitVal = arma::median( vals );
        //check if the median actually divides the points, if not use the midpoint
        double maxv = arma::max(vals);
        if (maxv == splitVal) {
            double minv = arma::min(vals);
            splitVal = (maxv+minv)/2.;
        }

        //create the update child bounds
        arma::Col<double> leftBounds = UpperBounds;
        leftBounds[splitDim] = splitVal;
        arma::Col<double> rightBounds = LowerBounds;
        rightBounds[splitDim] = splitVal;

        //separate the left and right child nodes
        std::vector<size_t> leftChildren, rightChildren, leftSpill, rightSpill;
        leftChildren.reserve(dataIndices.size()/2+1);
        rightChildren.reserve(dataIndices.size()/2+1);
        for (size_t i = 0; i < vals.n_elem; ++i) {
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

        //push back spill data from parent spill data
        for (const auto& c : spillIndices) {
            if (data[c].value()[splitDim] + spillEps > splitVal) {
                rightSpill.push_back(c);
            }
            if (data[c].value()[splitDim] - spillEps < splitVal) {
                leftSpill.push_back(c);
            }
        }

        size_t currentIndex = nodes.size();
        //make a new node with children
        nodes.push_back(KDTreeNode({
                parentIndex,
                splitDim,
                splitVal,
                maxRadius,
                LowerBounds,
                UpperBounds,
                std::vector<size_t>(),
                std::vector<size_t>(),
                mkNodeRecursive(currentIndex,
                                leftChildren,
                                leftSpill,
                                LowerBounds,
                                leftBounds),
                mkNodeRecursive(currentIndex,
                                rightChildren,
                                rightSpill,
                                rightBounds,
                                UpperBounds)
              }));
        return nodes.size()-1;
    };

public:
    typedef size_t nodeReturnType;

    ///create a KDTree with a dataset d, and a default maximum leaf-node size of 100 children
    KDTree(std::vector<OBJ>& d,
           const double& spillEps = 0.0,
           const size_t& mLeafSize = 100) : data(d), spillEps(spillEps) {
        maxLeafSize = mLeafSize;
        dataNodes.resize(data.size());
        nodes.reserve(data.capacity());
        std::vector<size_t> dataIndices;
        dataIndices.resize(data.size());
        arma::Col<double> lowerBounds = arma::conv_to<arma::Col<double>>::from(d[0].value());
        arma::Col<double> upperBounds = arma::conv_to<arma::Col<double>>::from(d[0].value());
        dataIndices[0] = 0;
        for (size_t i = 1; i < dataIndices.size(); ++i) {
            dataIndices[i] = i;
            lowerBounds = arma::min(lowerBounds, arma::conv_to<arma::Col<double>>::from(d[i].value()));
            upperBounds = arma::max(upperBounds, arma::conv_to<arma::Col<double>>::from(d[i].value()));
        }
        root = mkNodeRecursive(0,
                               dataIndices,
                               std::vector<size_t>(),
                               lowerBounds,
                               upperBounds);
    }

    ///Update the tree max-radius values for the item at dataIndex, which used to have the value oldRadius
    void updateRadius(const size_t& dataIndex, const double& oldRadius) {
        //let the tree know a data item radius has been updated, and propagate the changes through the tree.
        size_t currentNode = dataNodes[dataIndex];
        if (data[dataIndex].radius() > nodes[currentNode].maxRadius) { //this is the case where we are a new maxradius
            while (nodes[currentNode].maxRadius < data[dataIndex].radius()) {
                nodes[currentNode].maxRadius = data[dataIndex].radius();
                currentNode = nodes[currentNode].parentIndex;
                if (nodes[currentNode].parentIndex == currentNode) {
                    break;
                }
            }
        } else {
            while (nodes[currentNode].maxRadius == oldRadius) { //this is the case where we are an old maxradius
                if (nodes[currentNode].childData.size() > 0) {
                    nodes[currentNode].maxRadius = 0.0;
                    for (const auto& c: nodes[currentNode].childData) {
                        nodes[currentNode].maxRadius = std::max(
                                            nodes[currentNode].maxRadius,
                                            data[c].radius());
                    }
                } else {
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
                                                    const size_t& maxLeaves = std::numeric_limits<size_t>::max()) const {
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
               and std::get<0>(candidateHeap[0]) <= std::get<0>(neighbourHeap[0])*std::get<0>(neighbourHeap[0])
               and leafCounter < maxLeaves) {

            //get a new candidate KD-tree node
            std::pop_heap(candidateHeap.begin(),
                          candidateHeap.end(),
                          candidateCmp);
            auto current = candidateHeap.back();
            candidateHeap.resize(candidateHeap.size()-1);


            while (nodes[std::get<1>(current)].childData.size() == 0) {
                const KDTreeNode& cnode = nodes[std::get<1>(current)];
                //descend until we reach a leaf,
                const double splitDim = val.value()[cnode.splitDim];
                const double d = cnode.splitVal - splitDim;
                //note: this is a horrible approximation to avoid extra calculation for the real distance
                //it also applies only for L2 distances
                const double pDist = std::get<0>(current) + d*d;
                size_t farChild;
                if (splitDim > cnode.splitVal) {
                    farChild = cnode.leftChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.rightChildNode);
                } else {
                    farChild = cnode.rightChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.leftChildNode);
                }

                //put the more distant decision node onto the heap for later
                candidateHeap.push_back(std::make_pair(pDist, farChild));
                std::push_heap(candidateHeap.begin(),
                               candidateHeap.end(),
                               candidateCmp);
                }

            //add any near neighbours to the result
            const KDTreeNode& cnode = nodes[std::get<1>(current)];
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
                    and nodes[std::get<1>(current)].maxRadius * nodes[std::get<1>(current)].maxRadius >= std::get<0>(current)
                    ) {
                const KDTreeNode& cnode = nodes[std::get<1>(current)];
                //descend until we reach a leaf,
                const double splitDim = val.value()[cnode.splitDim];
                const double d = cnode.splitVal - splitDim;
                //note: this is a horrible approximation to avoid extra calculation for the real distance
                //it also applies only for L2 distances
                const double pDist = std::get<0>(current) + d*d;

                size_t farChild;
                if (splitDim > cnode.splitVal) {
                    farChild = cnode.leftChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.rightChildNode);
                } else {
                    farChild = cnode.rightChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.leftChildNode);
                }

                //put the more distant decision node onto the heap for later
                if (nodes[farChild].maxRadius * nodes[farChild].maxRadius >= pDist) {
                    candidateHeap.push_back(std::make_pair(pDist, farChild));
                }
            }
            //add any near neighbours to the result
            const KDTreeNode& cnode = nodes[std::get<1>(current)];
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

        //initialise heap comparisons
        auto candidateCmp = ([](const std::tuple<double,size_t>& a,
            const std::tuple<double,size_t>& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

        size_t leafCounter = 0;
        //keep going until the closest KD-tree node is further than the furthest neighbour
        while (candidateHeap.size() > 0
               and std::get<0>(candidateHeap[0]) < eps * eps
               and leafCounter < maxLeaves) {

            //get a new candidate KD-tree node
            std::pop_heap(candidateHeap.begin(),
                          candidateHeap.end(),
                          candidateCmp);
            auto current = candidateHeap.back();
            candidateHeap.resize(candidateHeap.size()-1);


            while (nodes[std::get<1>(current)].childData.size() == 0) {
                const KDTreeNode& cnode = nodes[std::get<1>(current)];
                //descend until we reach a leaf,
                const double splitDim = val.value()[cnode.splitDim];
                const double d = cnode.splitVal - splitDim;
                //note: this is a horrible approximation to avoid extra calculation for the real distance
                //it also applies only for L2 distances
                const double pDist = std::get<0>(current) + d*d;
                size_t farChild;
                if (splitDim > cnode.splitVal) {
                    farChild = cnode.leftChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.rightChildNode);
                } else {
                    farChild = cnode.rightChildNode;
                    current = std::make_tuple(std::get<0>(current), cnode.leftChildNode);
                }

                //put the more distant decision node onto the heap for later
                candidateHeap.push_back(std::make_pair(pDist, farChild));
                std::push_heap(candidateHeap.begin(),
                               candidateHeap.end(),
                               candidateCmp);
                }

            //add any near neighbours to the result
            const KDTreeNode& cnode = nodes[std::get<1>(current)];
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
        const arma::Col<double> val = arma::conv_to<arma::Col<double>>::from(item.value());
        if (node > nodes.size()) {
            node = root;
            while (nodes[node].childData.size() == 0) {
                if (val[nodes[node].splitDim] <= nodes[node].splitVal) {
                    node = nodes[node].leftChildNode;
                } else {
                    node = nodes[node].rightChildNode;
                }
            }
        }

        //propagate any new upper/lower bound limits up to the root node
        if (arma::any(val > nodes[node].UpperBounds)) {
            size_t current = node;
            nodes[current].UpperBounds = arma::max(nodes[current].UpperBounds, val);
            while (current != nodes[current].parentIndex) {
                current = nodes[current].parentIndex;
                nodes[current].UpperBounds = arma::max(nodes[current].UpperBounds, val);
            }
        }
        if (arma::any(val < nodes[node].LowerBounds)) {
            size_t current = node;
            nodes[current].LowerBounds = arma::min(nodes[current].LowerBounds, val);
            while (current != nodes[current].parentIndex) {
                current = nodes[current].parentIndex;
                nodes[current].LowerBounds = arma::min(nodes[current].LowerBounds, val);
            }
        }
        //insert item
        nodes[node].childData.push_back(data.size());
        data.push_back(item);
        dataNodes.push_back(node);

        //update the tree constraints
        updateRadius(data.size()-1, 0.0);
        splitNode(node);
    }

    //TODO: maybe delete and knn-renn query? Optional at this stage.
    //TODO2: fix distance estimation to be more awesome.
};

#endif

