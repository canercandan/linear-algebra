###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

LINK_LIBRARIES(${PROJECT_NAME})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

INCLUDE(${CMAKE_MODULE_PATH}/UseCUDA.cmake)

CUDA_INCLUDE_DIRECTORIES(${CORE_LIBRARY_DIR}/src)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

FILE(GLOB SOURCES "t-*.cu")

FOREACH(current ${SOURCES})
  GET_FILENAME_COMPONENT(basename "${current}" NAME_WE)
  CUDA_ADD_EXECUTABLE(${basename} ${current})
  GET_FILENAME_COMPONENT(basename "${current}" NAME_WE) # bugfix
  ADD_TEST(${basename} ${basename})
  CUDA_ADD_CUBLAS_TO_TARGET(${basename})
  CUDA_ADD_CUSPARSE_TO_TARGET(${basename})
  INSTALL(TARGETS ${basename} RUNTIME DESTINATION share/${PROJECT_NAME}/test COMPONENT test)
ENDFOREACH()

###############################################################################
