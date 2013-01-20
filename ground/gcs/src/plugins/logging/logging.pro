TEMPLATE = lib
TARGET = LoggingGadget
DEFINES += LOGGING_LIBRARY
QT += svg
include(../../taulabsgcsplugin.pri)
include(logging_dependencies.pri)
HEADERS += loggingplugin.h \
    logfile.h \
    logginggadgetwidget.h \
    logginggadget.h \
    logginggadgetfactory.h \
    loggingdevice.h
#    logginggadgetconfiguration.h
#   logginggadgetoptionspage.h

SOURCES += loggingplugin.cpp \
    logfile.cpp \
    logginggadgetwidget.cpp \
    logginggadget.cpp \
    logginggadgetfactory.cpp \
    loggingdevice.cpp
#    logginggadgetconfiguration.cpp \
#    logginggadgetoptionspage.cpp
OTHER_FILES += LoggingGadget.pluginspec
FORMS += logging.ui
#    logginggadgetwidget.ui \
#    loggingdialog.ui
