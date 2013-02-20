TEMPLATE = lib
TARGET = ScopeGadget
DEFINES += SCOPE_LIBRARY
include(../../taulabsgcsplugin.pri)
include (scope_dependencies.pri)
HEADERS += scopeplugin.h \
    scopes2d/scatterplotdata.h \
    scopes2d/scatterplotscopeconfig.h \
    scopes2d/histogramdata.h \
    scopes2d/histogramscopeconfig.h \
    scopes2d/scopes2dconfig.h \
    scopes3d/spectrogramdata.h \
    scopes3d/spectrogramscopeconfig.h \
    scopes3d/scopes3dconfig.h \
    scopesconfig.h \
    plotdata.h \
    plotdata2d.h \
    plotdata3d.h \
    scope_global.h
HEADERS += scopegadgetoptionspage.h
HEADERS += scopegadgetconfiguration.h
HEADERS += scopegadget.h
HEADERS += scopegadgetwidget.h
HEADERS += scopegadgetfactory.h
SOURCES += scopeplugin.cpp \
    scopes2d/histogramscopeconfig.cpp \
    scopes2d/scatterplotscopeconfig.cpp \
    scopes3d/spectrogramscopeconfig.cpp \
    plotdata.cpp
SOURCES += scopegadgetoptionspage.cpp
SOURCES += scopegadgetconfiguration.cpp
SOURCES += scopegadget.cpp
SOURCES += scopegadgetfactory.cpp
SOURCES += scopegadgetwidget.cpp
OTHER_FILES += ScopeGadget.pluginspec
FORMS += scopegadgetoptionspage.ui
