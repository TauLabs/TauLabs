TEMPLATE = lib 
TARGET = SetupWizard 
QT += svg


include(../../taulabsgcsplugin.pri)
include(../../plugins/coreplugin/coreplugin.pri)
include(../../plugins/uavobjectutil/uavobjectutil.pri)
include(../../plugins/uavobjectwidgetutils/uavobjectwidgetutils.pri)
include(../../plugins/config/config.pri)

LIBS *= -l$$qtLibraryName(Uploader)
HEADERS += setupwizardplugin.h \ 
    setupwizard.h \
    pages/boardtype_unknown.h \
    pages/controllerpage.h \
    pages/vehiclepage.h \
    pages/notyetimplementedpage.h \
    pages/multipage.h \
    pages/fixedwingpage.h \
    pages/helipage.h \
    pages/surfacepage.h \
    pages/abstractwizardpage.h \
    pages/outputpage.h \
    pages/inputpage.h \
    pages/inputpage_notsupported.h \
    pages/summarypage.h \
    vehicleconfigurationsource.h \
    vehicleconfigurationhelper.h \
    connectiondiagram.h \
    pages/outputcalibrationpage.h \
    outputcalibrationutil.h \
    pages/rebootpage.h \
    pages/savepage.h \
    pages/autoupdatepage.h \
    pages/biascalibrationpage.h \
    pages/tlendpage.h \
    pages/tlstartpage.h

SOURCES += setupwizardplugin.cpp \
    setupwizard.cpp \
    pages/boardtype_unknown.cpp \
    pages/controllerpage.cpp \
    pages/vehiclepage.cpp \
    pages/notyetimplementedpage.cpp \
    pages/multipage.cpp \
    pages/fixedwingpage.cpp \
    pages/helipage.cpp \
    pages/surfacepage.cpp \
    pages/abstractwizardpage.cpp \
    pages/outputpage.cpp \
    pages/inputpage.cpp \
    pages/inputpage_notsupported.cpp \
    pages/summarypage.cpp \
    vehicleconfigurationsource.cpp \
    vehicleconfigurationhelper.cpp \
    connectiondiagram.cpp \
    pages/outputcalibrationpage.cpp \
    outputcalibrationutil.cpp \
    pages/rebootpage.cpp \
    pages/savepage.cpp \
    pages/autoupdatepage.cpp \
    pages/biascalibrationpage.cpp \
    pages/tlendpage.cpp \
    pages/tlstartpage.cpp

OTHER_FILES += SetupWizard.pluginspec \
    SetupWizard.json

FORMS += \
    pages/startpage.ui \
    pages/endpage.ui \
    pages/boardtype_unknown.ui \
    pages/controllerpage.ui \
    pages/vehiclepage.ui \
    pages/notyetimplementedpage.ui \
    pages/multipage.ui \
    pages/fixedwingpage.ui \
    pages/helipage.ui \
    pages/surfacepage.ui \
    pages/outputpage.ui \
    pages/inputpage.ui \
    pages/inputpage_notsupported.ui \
    pages/summarypage.ui \
    connectiondiagram.ui \
    pages/outputcalibrationpage.ui \
    pages/rebootpage.ui \
    pages/savepage.ui \
    pages/autoupdatepage.ui \
    pages/biascalibrationpage.ui

RESOURCES += \
    wizardResources.qrc
