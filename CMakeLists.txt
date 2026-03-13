cmake_minimum_required(VERSION 3.26.5)
project(decision_tree)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fno-math-errno -fno-trapping-math -freciprocal-math")

find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI QUIET)

add_library(libs
        src/Dataset.cpp
        src/RandomForestClassifier.cpp
        src/DecisionTreeClassifier.cpp
        src/Timer.cpp
        include/splitters/SplitterMP.hpp
)

target_include_directories(libs PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/pcg32
        ${CMAKE_SOURCE_DIR}/spdlog
)

target_link_libraries(libs PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(libs PUBLIC TBB::tbb)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE libs)

if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_C_COMPILER}")
    target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
    target_compile_definitions(mpi PRIVATE MPI_AVAILABLE)
    target_link_libraries(libs PUBLIC MPI::MPI_CXX)
    target_compile_definitions(libs PUBLIC MPI_AVAILABLE)
else()
    message(STATUS "MPI not found, compiling mpi target without MPI support")
endif()
