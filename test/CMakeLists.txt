###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(SOURCES
  )

FOREACH(current ${SOURCES})
  ADD_EXECUTABLE(${current} ${current}.cpp ${COMMON_SOURCES})
  ADD_TEST(${current} ${current})
  TARGET_LINK_LIBRARIES(${current} ${PROJECT_NAME})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/${PROJECT_NAME}/test COMPONENT test)
ENDFOREACH()

######################################################################################
