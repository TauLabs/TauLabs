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
SDL {
SUBDIRS += sdlgamepad
}

!LIGHTWEIGHT_GCS {
    SUBDIRS += glc_lib
}

SUBDIRS +=
