###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

LINK_LIBRARIES(${PROJECT_NAME})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

ADD_SUBDIRECTORY(device)
ADD_SUBDIRECTORY(host)

###############################################################################
