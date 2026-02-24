add_library(my_lib
    NeuralNetwork.cpp 
    Neuron.cpp 
    Layer.cpp 
    MathUtility.cpp 
    Matrix.cpp
)

target_include_directories(my_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(my_lib PUBLIC cxx_std_17)