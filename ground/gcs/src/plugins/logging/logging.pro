TEMPLATE = lib
TARGET = LoggingGadget
DEFINES += LOGGING_LIBRARY
QT += xml

include(../../taulabsgcsplugin.pri)
include(logging_dependencies.pri)

INCLUDEPATH *= ../../libs/glc_lib

HEADERS += loggingplugin.h \
    logfile.h \
    logginggadgetwidget.h \
    logginggadget.h \
    logginggadgetfactory.h \
    loggingdevice.h

SOURCES += loggingplugin.cpp \
    logfile.cpp \
    logginggadgetwidget.cpp \
    logginggadget.cpp \
    logginggadgetfactory.cpp \
    loggingdevice.cpp

OTHER_FILES += LoggingGadget.pluginspec
FORMS += logging.ui
