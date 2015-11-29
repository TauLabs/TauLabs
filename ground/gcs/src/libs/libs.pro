TEMPLATE  = subdirs
CONFIG   += ordered
QT += widgets
SUBDIRS   = \
    qscispinbox\
    qtconcurrent \
    aggregation \
    extensionsystem \
    utils \
    tlmapcontrol \
    qwt \
    libcrashreporter-qt
win32 {
SUBDIRS   += \
    zlib
}
SUBDIRS   += \
    quazip
SDL {
SUBDIRS += sdlgamepad
}

!LIGHTWEIGHT_GCS {
    SUBDIRS += glc_lib
}

SUBDIRS +=
