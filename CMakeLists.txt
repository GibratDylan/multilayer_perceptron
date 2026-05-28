cmake_minimum_required(VERSION 3.16)

project(mlp LANGUAGES CXX)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/include/Eigen/Core")
	message(STATUS "Downloading and extracting Eigen 3.4.0...")
	execute_process(
		COMMAND bash ${CMAKE_SOURCE_DIR}/tool/download_eigen.sh
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		RESULT_VARIABLE EIGEN_DOWNLOAD_RESULT
	)
	if(NOT EIGEN_DOWNLOAD_RESULT EQUAL 0)
		message(FATAL_ERROR "Failed to download Eigen")
	endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_C_COMPILER "/usr/bin/clang")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(SRCS_DIR src)
set(HEADERS_DIR include)
add_executable(${PROJECT_NAME}
	${SRCS_DIR}/main.cpp
	${SRCS_DIR}/NeuronalLayer.cpp
	${SRCS_DIR}/activation/AActivation.cpp
	${SRCS_DIR}/activation/ActivationReLU.cpp
	${SRCS_DIR}/activation/ActivationSoftmax.cpp
	${SRCS_DIR}/loss/ALoss.cpp
	${SRCS_DIR}/loss/LossCategoricalCrossEntropy.cpp
	${SRCS_DIR}/activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.cpp
	${SRCS_DIR}/trainer/Trainer.cpp
	${SRCS_DIR}/trainer/Metrics.cpp
	${SRCS_DIR}/trainer/observer/TrainerObserverMetricsWriter.cpp
	${SRCS_DIR}/network/Network.cpp
	${SRCS_DIR}/config/Config.cpp

	${HEADERS_DIR}/NeuronalLayer.hpp
	${HEADERS_DIR}/activation/AActivation.hpp
	${HEADERS_DIR}/activation/ActivationReLU.hpp
	${HEADERS_DIR}/activation/ActivationSoftmax.hpp
	${HEADERS_DIR}/loss/ALoss.hpp
	${HEADERS_DIR}/loss/LossCategoricalCrossEntropy.hpp
	${HEADERS_DIR}/activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp
	${HEADERS_DIR}/trainer/Trainer.hpp
	${HEADERS_DIR}/trainer/Metrics.hpp
	${HEADERS_DIR}/trainer/observer/TrainerObserverMetricsWriter.hpp
	${HEADERS_DIR}/trainer/observer/ITrainerObserver.hpp
	${HEADERS_DIR}/network/Network.hpp
	${HEADERS_DIR}/config/Config.hpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_SOURCE_DIR}/${HEADERS_DIR}
)

target_compile_options(${PROJECT_NAME} PRIVATE
	-Wall
	-Wextra
	-Werror
)

add_custom_target(install-libs
	COMMAND bash ${CMAKE_SOURCE_DIR}/tools/check_libs.sh
	COMMAND $<TARGET_FILE:${PROJECT_NAME}>
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Checking libs"
)

add_custom_target(run
	COMMAND $<TARGET_FILE:${PROJECT_NAME}>
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Building and running ${PROJECT_NAME}"
)

add_custom_target(tidy-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tool/clang_tidy.sh --all
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Checking lint error with clang-tidy"
)

add_custom_target(tidy-fix-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tool/clang_tidy.sh --all --fix
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Fixing all files with with clang-tidy"
)

add_custom_target(format-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tool/clang_format.sh --all
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Checking formatting with clang-format"
)

add_custom_target(format-fix-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tool/clang_format.sh --all --fix
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Formatting all files with clang-format"
)