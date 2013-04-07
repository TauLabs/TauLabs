include(../gcs.pri)

isEmpty(PROVIDER) {
    PROVIDER = TauLabs
}
CONFIG       += plugin
QT += widgets
DESTDIR = $$GCS_PLUGIN_PATH/$$PROVIDER
LIBS += -L$$DESTDIR
INCLUDEPATH *= $$GCS_SOURCE_TREE/src/plugins
DEPENDPATH += $$GCS_SOURCE_TREE/src/plugins

# copy the plugin spec
isEmpty(TARGET) {
    error("taulabsgcsplugin.pri: You must provide a TARGET")
}

defineReplace(stripOutDir) {
    return($$relative_path($$1, $$OUT_PWD))
}

PLUGINSPEC = $$_PRO_FILE_PWD_/$${TARGET}.pluginspec
PLUGINSPEC_IN = $${PLUGINSPEC}.in
exists($$PLUGINSPEC_IN) {
    OTHER_FILES += $$PLUGINSPEC_IN
    QMAKE_SUBSTITUTES += $$PLUGINSPEC_IN
    PLUGINSPEC = $$OUT_PWD/$${TARGET}.pluginspec
    copy2build.output = $$DESTDIR/${QMAKE_FUNC_FILE_IN_stripOutDir}
} else {
    # need to support that for external plugins
    OTHER_FILES += $$PLUGINSPEC
    copy2build.output = $$DESTDIR/${QMAKE_FUNC_FILE_IN_stripSrcDir}
}
copy2build.input = PLUGINSPEC
isEmpty(vcproj):copy2build.variable_out = PRE_TARGETDEPS
copy2build.commands = $$QMAKE_COPY ${QMAKE_FILE_IN} ${QMAKE_FILE_OUT}
copy2build.name = COPY ${QMAKE_FILE_IN}
copy2build.CONFIG += no_link
QMAKE_EXTRA_COMPILERS += copy2build

TARGET = $$qtLibraryName($$TARGET)

macx {
        QMAKE_LFLAGS_SONAME = -Wl,-install_name,@executable_path/../Plugins/$${PROVIDER}/
} else:linux-* {
    #do the rpath by hand since it's not possible to use ORIGIN in QMAKE_RPATHDIR
    QMAKE_RPATHDIR += \$\$ORIGIN
    QMAKE_RPATHDIR += \$\$ORIGIN/..
    QMAKE_RPATHDIR += \$\$ORIGIN/../..
    GCS_PLUGIN_RPATH = $$join(QMAKE_RPATHDIR, ":")
    QMAKE_LFLAGS += -Wl,-z,origin \'-Wl,-rpath,$${GCS_PLUGIN_RPATH}\'
    QMAKE_RPATHDIR =
}


contains(QT_CONFIG, reduce_exports):CONFIG += hide_symbols

CONFIG += plugin plugin_with_soname
linux*:QMAKE_LFLAGS += $$QMAKE_LFLAGS_NOUNDEF

!macx {
    target.path = /$$GCS_LIBRARY_BASENAME/taulabs/plugins/$$PROVIDER
    pluginspec.files += $${TARGET}.pluginspec
    pluginspec.path = /$$GCS_LIBRARY_BASENAME/taulabs/plugins/$$PROVIDER
    INSTALLS += target pluginspec
}
