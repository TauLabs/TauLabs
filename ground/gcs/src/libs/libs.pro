TEMPLATE  = subdirs
CONFIG   += ordered

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
