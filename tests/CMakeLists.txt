include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(neural_network_tests
    "MatrixTest.cpp"
    "MathUtilityTest.cpp" "NeuronTest.cpp" "LayerTest.cpp" "NeuralNetworkTest.cpp")

target_link_libraries(neural_network_tests
    PRIVATE
        my_lib
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(neural_network_tests)