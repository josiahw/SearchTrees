add_library(covertree CoverTree.h CoverTree.cpp)


target_link_libraries(covertree
                    ${ARMADILLO_LIBRARIES})