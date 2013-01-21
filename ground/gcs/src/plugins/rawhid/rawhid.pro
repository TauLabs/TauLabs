TEMPLATE = lib
TARGET = RawHID
include(../../taulabsgcsplugin.pri)
include(rawhid_dependencies.pri)
HEADERS += rawhid_global.h \
    rawhidplugin.h \
    rawhid.h \
    pjrc_rawhid.h \
    rawhid_const.h \
    usbmonitor.h \
    usbsignalfilter.h \
    usbdevice.h
SOURCES += rawhidplugin.cpp \
    rawhid.cpp \
    usbsignalfilter.cpp \
    usbdevice.cpp
FORMS += 
RESOURCES += 
DEFINES += RAWHID_LIBRARY
OTHER_FILES += RawHID.pluginspec

# Platform Specific USB HID Stuff
win32 { 
    SOURCES += pjrc_rawhid_win.cpp \
        usbmonitor_win.cpp
    LIBS += -lhid \
        -lsetupapi
}
macx { 
    SOURCES += pjrc_rawhid_mac.cpp \
            usbmonitor_mac.cpp
    LIBS += -framework IOKit \
        -framework CoreFoundation
}
linux-g++ {
    SOURCES += pjrc_rawhid_unix.cpp \
            usbmonitor_linux.cpp
    LIBS += -lusb-1.0 -ludev
}
linux-g++-64 {
    SOURCES += pjrc_rawhid_unix.cpp \
            usbmonitor_linux.cpp
    LIBS += -lusb-1.0 -ludev
}
