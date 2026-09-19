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

add_executable(Neural_Network_From_Scratch main.cpp)
target_link_libraries(Neural_Network_From_Scratch PRIVATE neural_network_lib)

if(MSVC)
    target_compile_options(neural_network_lib PRIVATE /O2 /arch:AVX2 /fp:fast /Zi)
    target_compile_options(Neural_Network_From_Scratch PRIVATE /O2 /arch:AVX2 /fp:fast /Zi)
    target_link_options(Neural_Network_From_Scratch PRIVATE /DEBUG)
else()
    target_compile_options(neural_network_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(Neural_Network_From_Scratch PRIVATE -O3 -march=native -ffast-math)
endif()