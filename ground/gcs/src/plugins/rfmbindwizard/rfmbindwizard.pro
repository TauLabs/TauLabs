TEMPLATE = lib 
TARGET = RfmBindWizard
QT += svg


include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)
include(../../plugins/uavobjectwidgetutils/uavobjectwidgetutils.pri)
include(../../plugins/config/config.pri)

LIBS *= -l$$qtLibraryName(Uploader)
HEADERS += rfmbindwizardplugin.h \ 
    rfmbindwizard.h \
    pages/abstractwizardpage.h \
    pages/tlendpage.h \
    pages/tlstartpage.h \
    pages/coordinatorpage.h

SOURCES += rfmbindwizardplugin.cpp \
    rfmbindwizard.cpp \
    pages/abstractwizardpage.cpp \
    pages/tlendpage.cpp \
    pages/tlstartpage.cpp \
    pages/coordinatorpage.cpp

OTHER_FILES += RfmBindWizard.pluginspec \
    RfmBindWizard.json

FORMS += \
    pages/startpage.ui \
    pages/endpage.ui \
    pages/coordinatorpage.ui

RESOURCES += \
    rfmbindResources.qrc
