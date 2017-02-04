# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

# Project's name

project(SearchTrees)
# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O3 -flto -march=native -mtune=native")




#find libraries
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#check C++11 support:
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -ggdb")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#check openmp support
CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPILER_SUPPORTS_OPENMP)
if (COMPILER_SUPPORTS_OPENMP)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/bruteforce")
add_subdirectory(bruteforce)
include_directories("${PROJECT_SOURCE_DIR}/kdtree")
add_subdirectory(kdtree)
include_directories("${PROJECT_SOURCE_DIR}/balltree")
add_subdirectory(balltree)
include_directories("${PROJECT_SOURCE_DIR}/covertree")
add_subdirectory(covertree)
include_directories("${PROJECT_SOURCE_DIR}/testing")
add_subdirectory(testing)
include_directories("${PROJECT_SOURCE_DIR}/thesisexperiments")
add_subdirectory(thesisexperiments)

#create executables
add_executable(UnitTests ${PROJECT_SOURCE_DIR}/UnitTests.cpp)
add_executable(ThesisDataSizeTests ${PROJECT_SOURCE_DIR}/ThesisDataSizeTests.cpp)
add_executable(ThesisLeafLimitTests ${PROJECT_SOURCE_DIR}/ThesisLeafLimitTests.cpp)
add_executable(ThesisSpillTreeTests ${PROJECT_SOURCE_DIR}/ThesisSpillTreeTests.cpp)

target_link_libraries(UnitTests
                    euclideannode
                    euclideanknnnode
                    bruteforce
                    kdtree
                    balltree
                    covertree
                    testtree
                    ${ARMADILLO_LIBRARIES})

target_link_libraries(ThesisDataSizeTests
                    datareferenceeuclideannode
                    bruteforce
                    kdtree
                    balltree
                    covertree
                    thesistest
                    ${ARMADILLO_LIBRARIES})

target_link_libraries(ThesisLeafLimitTests
                    datareferenceeuclideannode
                    bruteforce
                    kdtree
                    balltree
                    covertree
                    thesistest
                    ${ARMADILLO_LIBRARIES})

target_link_libraries(ThesisSpillTreeTests
                    datareferenceeuclideannode
                    bruteforce
                    kdtree
                    balltree
                    covertree
                    thesistest
                    ${ARMADILLO_LIBRARIES})
