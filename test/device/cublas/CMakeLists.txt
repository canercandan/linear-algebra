SET(SOURCES
  t-create-matrix
  t-docexample
  t-gemv
  )

FOREACH(current ${SOURCES})
  CUDA_ADD_EXECUTABLE(${current} ${current}.cu)
  ADD_TEST(${current} ${current})
  CUDA_ADD_CUBLAS_TO_TARGET(${current})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/${PROJECT_NAME}/test/device/cublas COMPONENT test)
ENDFOREACH()
