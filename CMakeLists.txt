cmake_minimum_required(VERSION 3.15)
project(Neural_Network_From_Scratch)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(
	Neural_Network_From_Scratch ${SOURCES} 
	"src/NeuralNetwork.cpp" 
	"include/NeuralNetwork.h" 
	"src/Neuron.cpp" 
	"include/Neuron.h" 
	"src/Layer.cpp" 
	"include/Layer.h" 
	"src/MathUtility.cpp" 
	"include/MathUtility.h" 
	"include/Matrix.h"
	"src/Matrix.cpp"
)
