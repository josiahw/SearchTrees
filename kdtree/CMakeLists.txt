add_library(kdtree KDTree.cpp KDTree.h)


target_link_libraries(kdtree
                    ${ARMADILLO_LIBRARIES})