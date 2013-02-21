TEMPLATE  = subdirs
CONFIG   += ordered
QT += widgets
SUBDIRS   = \
    qscispinbox\
    qtconcurrent \
    aggregation \
    extensionsystem \
    utils \
    opmapcontrol \
    qwt \
    qextserialport \
    sdlgamepad \
    libqxt

!LIGHTWEIGHT_GCS {
SUBDIRS += glc_lib
}

SUBDIRS +=
