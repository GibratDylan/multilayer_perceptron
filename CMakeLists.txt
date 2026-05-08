cmake_minimum_required(VERSION 3.16)

project(mlp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_C_COMPILER "/usr/bin/clang")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_executable(${PROJECT_NAME}
	srcs/main.cpp
	srcs/Layer.cpp
	srcs/activation_fonction/AActivation.cpp
	srcs/activation_fonction/ActivationReLU.cpp
	srcs/activation_fonction/ActivationSoftmax.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_SOURCE_DIR}/libs/eigen-3.4.0
)

target_compile_options(${PROJECT_NAME} PRIVATE
	-Wall
	-Wextra
	-Werror
)

add_custom_target(run
	COMMAND $<TARGET_FILE:${PROJECT_NAME}>
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Building and running ${PROJECT_NAME}"
)

add_custom_target(tidy-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --all
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Checking lint error with clang-tidy"
)
add_custom_target(tidy-fix-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --all --fix
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Fixing all C++ files with with clang-tidy"
)

add_custom_target(format-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_format.sh --all
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Checking formatting with clang-format"
)

add_custom_target(format-fix-all
	COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_format.sh --all --fix
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Formatting all C++ files with clang-format"
)