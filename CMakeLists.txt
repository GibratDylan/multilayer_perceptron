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
)

target_include_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_SOURCE_DIR}/libs/eigen-3.4.0
)

target_compile_options(${PROJECT_NAME} PRIVATE
	-Wall
	-Wextra
	-Werror
)

# Equivalent to: make run
add_custom_target(run
	COMMAND $<TARGET_FILE:${PROJECT_NAME}>
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Building and running ${PROJECT_NAME}"
)

# Equivalent to: make test-all (only if present)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/run_all.sh")
	add_custom_target(test-all
		COMMAND bash ${CMAKE_SOURCE_DIR}/tests/run_all.sh
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		DEPENDS ${PROJECT_NAME}
		COMMENT "Running all tests"
	)
endif()

# Equivalent to tidy targets (only if script exists)
if(EXISTS "${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh")
	add_custom_target(tidy-db
		COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --build-db
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
	add_custom_target(tidy
		COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --auto-build-db --changed
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
	add_custom_target(tidy-all
		COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --auto-build-db --all
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
	add_custom_target(tidy-fix
		COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --auto-build-db --changed --fix
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
	add_custom_target(tidy-fix-all
		COMMAND bash ${CMAKE_SOURCE_DIR}/tools/clang_tidy.sh --auto-build-db --all --fix
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
endif()

# Equivalent to format targets (if clang-format is available)
find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)
	file(GLOB_RECURSE CPP_FILES
		"${CMAKE_SOURCE_DIR}/*.cpp"
		"${CMAKE_SOURCE_DIR}/*.hpp"
	)

	add_custom_target(format-all
		COMMAND ${CLANG_FORMAT_EXE} --dry-run -Werror ${CPP_FILES}
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMENT "Checking formatting with clang-format"
	)

	add_custom_target(format-fix-all
		COMMAND ${CLANG_FORMAT_EXE} -i ${CPP_FILES}
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMENT "Formatting all C++ files with clang-format"
	)
endif()