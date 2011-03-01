PROJECT(cuda)

FILE(GLOB SOURCES "t-*.cu")

MESSAGE(STATUS ${SOURCES})

FOREACH(current ${SOURCES})
  GET_FILENAME_COMPONENT(basename "${current}" NAME_WE)
  MESSAGE(STATUS ${basename})
  CUDA_ADD_EXECUTABLE(${PROJECT_NAME}-${basename} ${current})
  ADD_TEST(${PROJECT_NAME}-${basename} ${PROJECT_NAME}-${basename})
  GET_FILENAME_COMPONENT(basename "${current}" NAME_WE)
  CUDA_ADD_CUSPARSE_TO_TARGET(${PROJECT_NAME}-${basename})
  INSTALL(TARGETS ${PROJECT_NAME}-${basename} RUNTIME DESTINATION share/linear_algebra/test/device/${PROJECT_NAME} COMPONENT test)
ENDFOREACH()
