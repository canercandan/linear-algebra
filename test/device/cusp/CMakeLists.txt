PROJECT(cusp)

SET(SOURCES
  t-docexample
  t-matrix-csr
  t-matrix-coo
  t-matrix-ell
  t-matrix-dia
  t-matrix-hyb
  t-matvec
  t-cusp-matvec
  t-matvec-dense
  t-cusp-matvec-dense
  t-dot
  t-scal
  t-norm
  )

FOREACH(current ${SOURCES})
  CUDA_ADD_EXECUTABLE(${PROJECT_NAME}-${current} ${current}.cu)
  ADD_TEST(${PROJECT_NAME}-${current} ${PROJECT_NAME}-${current})
  CUDA_ADD_CUSPARSE_TO_TARGET(${PROJECT_NAME}-${current})
  INSTALL(TARGETS ${PROJECT_NAME}-${current} RUNTIME DESTINATION share/linear_algebra/test/device/${PROJECT_NAME} COMPONENT test)
ENDFOREACH()
