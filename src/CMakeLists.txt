add_library(neural_network_lib
    NeuralNetwork.cpp 
    Neuron.cpp 
    Layer.cpp 
    MathUtility.cpp 
    Matrix.cpp
    MNIST_dataloader.cpp
)

target_include_directories(neural_network_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(neural_network_lib PUBLIC cxx_std_17)