cmake_minimum_required(VERSION 3.20)
project(jneuro VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# spec/02: exact IEEE-754 arithmetic. Refuse any flag that relaxes it.
set(_jn_flag_vars CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
foreach(_v IN LISTS _jn_flag_vars)
  if("${${_v}}" MATCHES "fast-math|Ofast|unsafe-math-optimizations|finite-math-only|/fp:fast")
    message(FATAL_ERROR "${_v} contains a fast-math flag (${${_v}}); j_neuro requires exact IEEE-754 arithmetic (spec/02).")
  endif()
endforeach()

find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)

add_library(jneuro
  src/numerics.cpp
  src/rng.cpp
  src/types.cpp
  src/aggregate.cpp
  src/profile.cpp
  src/io.cpp
  src/sha256.cpp)
target_include_directories(jneuro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jneuro PUBLIC nlohmann_json::nlohmann_json Threads::Threads)
if(MSVC)
  target_compile_options(jneuro PUBLIC /fp:strict)
else()
  target_compile_options(jneuro PUBLIC -ffp-contract=off -fno-fast-math)
  target_compile_options(jneuro PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(JNEURO_BUILD_TESTS "Build the j_neuro test suite" ON)
if(JNEURO_BUILD_TESTS)
  find_package(Catch2 3 REQUIRED)
  include(CTest)
  include(Catch)
  set(JNEURO_TEST_SOURCES
    tests/test_numerics.cpp
    tests/test_rng.cpp
    tests/test_aggregate.cpp
    tests/test_eval.cpp
    tests/test_io_create.cpp
    tests/test_analysis.cpp
    tests/test_removal.cpp
    tests/test_mutate.cpp
    tests/test_v02.cpp
    tests/test_population.cpp
    CACHE STRING "Test sources to build (all by default)")
  add_executable(jneuro_tests ${JNEURO_TEST_SOURCES})
  target_link_libraries(jneuro_tests PRIVATE jneuro Catch2::Catch2WithMain)
  target_compile_definitions(jneuro_tests PRIVATE
    JNEURO_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/golden")
  catch_discover_tests(jneuro_tests)
endif()

option(JNEURO_BUILD_EXAMPLES "Build the example programs" ON)
if(JNEURO_BUILD_EXAMPLES)
  add_executable(blackjack_trainer ${CMAKE_CURRENT_SOURCE_DIR}/../examples/blackjack/blackjack.cpp)
  target_link_libraries(blackjack_trainer PRIVATE jneuro)
  add_executable(blackjack_analyze ${CMAKE_CURRENT_SOURCE_DIR}/../examples/blackjack/analyze.cpp)
  target_link_libraries(blackjack_analyze PRIVATE jneuro)
  add_executable(highlow_trainer ${CMAKE_CURRENT_SOURCE_DIR}/../examples/highlow/highlow.cpp)
  target_link_libraries(highlow_trainer PRIVATE jneuro)
  add_executable(expansion_test ${CMAKE_CURRENT_SOURCE_DIR}/../examples/expansion/expansion.cpp)
  target_link_libraries(expansion_test PRIVATE jneuro)
endif()
