add_library(bruteforce BruteForce.cpp BruteForce.h)


target_link_libraries(bruteforce
                    ${ARMADILLO_LIBRARIES})