TEMPLATE = lib
QT+=widgets
TARGET = ScopeGadget
DEFINES += SCOPE_LIBRARY
include(../../taulabsgcsplugin.pri)
include (scope_dependencies.pri)
HEADERS += scopeplugin.h \
    scopes2d/histogramplotdata.h \
    scopes2d/histogramscopeconfig.h \
    scopes2d/scatterplotdata.h \
    scopes2d/scatterplotscopeconfig.h \
    scopes3d/spectrogramplotdata.h \
    scopes3d/spectrogramscopeconfig.h \
    scopes2d/plotdata2d.h \
    scopes2d/scopes2dconfig.h \
    scopes3d/plotdata3d.h \
    scopes3d/scopes3dconfig.h \
    scopesconfig.h \
    plotdata.h \
    scope_global.h
HEADERS += scopegadgetoptionspage.h
HEADERS += scopegadgetconfiguration.h
HEADERS += scopegadget.h
HEADERS += scopegadgetwidget.h
HEADERS += scopegadgetfactory.h
SOURCES += scopeplugin.cpp \
    scopes2d/histogramplotdata.cpp \
    scopes2d/histogramscopeconfig.cpp \
    scopes2d/scatterplotdata.cpp \
    scopes2d/scatterplotscopeconfig.cpp \
    scopes3d/spectrogramplotdata.cpp \
    scopes3d/spectrogramscopeconfig.cpp \
    plotdata.cpp
SOURCES += scopegadgetoptionspage.cpp
SOURCES += scopegadgetconfiguration.cpp
SOURCES += scopegadget.cpp
SOURCES += scopegadgetfactory.cpp
SOURCES += scopegadgetwidget.cpp
OTHER_FILES += ScopeGadget.pluginspec \
    ScopeGadget.json
FORMS += scopegadgetoptionspage.ui
