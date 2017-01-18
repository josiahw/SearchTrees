add_library(balltree BallTree.h BallTree.cpp)


target_link_libraries(balltree
                    ${ARMADILLO_LIBRARIES})