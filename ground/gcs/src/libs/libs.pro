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
    qextserialport \
    libqxt

SDL {
SUBDIRS += sdlgamepad
}

#SUBDIRS += glc_lib
SUBDIRS +=
