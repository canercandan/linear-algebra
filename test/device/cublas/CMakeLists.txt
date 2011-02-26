PROJECT(cublas)

SET(SOURCES
  t-matvec
  t-docexample
  t-gemv
  cublas-t-powermethod
  )

FOREACH(current ${SOURCES})
  CUDA_ADD_EXECUTABLE(${PROJECT_NAME}-${current} ${current}.cu)
  ADD_TEST(${PROJECT_NAME}-${current} ${PROJECT_NAME}-${current})
  CUDA_ADD_CUBLAS_TO_TARGET(${PROJECT_NAME}-${current})
  INSTALL(TARGETS ${PROJECT_NAME}-${current} RUNTIME DESTINATION share/linear_algebra/test/device/${PROJECT_NAME} COMPONENT test)
ENDFOREACH()
