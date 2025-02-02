#ifndef INITMYPLUGIN_H
#define INITMYPLUGIN_H

#include <sofa/helper/config.h>

#ifdef SOFA_BUILD_MYPLUGIN
#define SOFA_MYPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_MYPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

/** mainpage
This is the main page of the doxygen documentation for MyPlugin.
 */

#endif