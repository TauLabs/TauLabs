
TEMPLATE = lib 
TARGET = NavWizard
QT += svg


include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)
include(../../plugins/config/config.pri)

LIBS *= -l$$qtLibraryName(Uploader)
HEADERS += navwizardplugin.h \
    pages/startpage.h \
    pages/endpage.h \
    pages/boardtype_unknown.h \
    pages/notyetimplementedpage.h \
    pages/failsafepage.h \
    pages/abstractwizardpage.h \
    pages/summarypage.h \
    vehicleconfigurationsource.h \
    vehicleconfigurationhelper.h \
    pages/rebootpage.h \
    pages/savepage.h \
    navigationwizard.h

SOURCES += navwizardplugin.cpp \
    pages/startpage.cpp \
    pages/endpage.cpp \
    pages/boardtype_unknown.cpp \
    pages/notyetimplementedpage.cpp \
    pages/failsafepage.cpp \
    pages/abstractwizardpage.cpp \
    pages/summarypage.cpp \
    vehicleconfigurationsource.cpp \
    vehicleconfigurationhelper.cpp \
    pages/rebootpage.cpp \
    pages/savepage.cpp \
    navigationwizard.cpp

OTHER_FILES += NavWizard.pluginspec

FORMS += \
    pages/startpage.ui \
    pages/endpage.ui \
    pages/boardtype_unknown.ui \
    pages/notyetimplementedpage.ui \
    pages/failsafepage.ui \
    pages/summarypage.ui \
    pages/rebootpage.ui \
    pages/savepage.ui

RESOURCES += \
    wizardResources.qrc
