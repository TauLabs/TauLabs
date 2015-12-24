include(uavobjectwidgetutils_dependencies.pri)

# Add the include path to the built-in uavobject include files.
INCLUDEPATH *= $$PWD
DEPENDPATH *= $$PWD

LIBS *= -l$$qtLibraryName(UAVObjectWidgetUtils)
