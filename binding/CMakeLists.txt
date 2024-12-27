project(MyPluginBindings)

if(NOT TARGET SofaPython3::Plugin)
    find_package(SofaPython3 REQUIRED COMPONENTS Plugin Bindings.Sofa)
endif()

SP3_add_python_module(
    TARGET ${PROJECT_NAME}
    PACKAGE Bindings.Modules
    MODULE MyPlugin
    DESTINATION Sofa
    SOURCES ${SOURCE_FILES}
    HEADERS ${HEADER_FILES}
    DEPENDS SofaPython3::Plugin SofaPython3::Bindings.Sofa MyPlugin
)
message("-- SofaPython3 bindings for MyPlugin will be created.")