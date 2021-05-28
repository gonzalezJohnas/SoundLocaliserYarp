#Copyright: (C) 2018 RobotCub Consortium
# Authors: Gonzalez Jonas
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

set(appname ${KEYWORD})

file(GLOB conf ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.ini ${CMAKE_CURRENT_SOURCE_DIR}/conf/*/*.pb ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.png)
file(GLOB scripts ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.template)


yarp_install(FILES ${conf} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})
