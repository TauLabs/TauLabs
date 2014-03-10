include(gcs.pri)

TEMPLATE = subdirs
# Copy Qt runtime libraries into the build directory (to run or package)
equals(copydata, 1) {
    # Windows release only, no debug target DLLs ending with 'd'
    win32:CONFIG(release, debug|release) {

        # copy Qt DLLs and phonon4
        QT_DLLS = Qt5Core.dll \
            Qt5Gui.dll \
            Qt5Network.dll \
            Qt5OpenGL.dll \
            Qt5Sql.dll \
            Qt5Svg.dll \
            Qt5Test.dll \
            Qt5Xml.dll \
            Qt5Declarative.dll \
            Qt5XmlPatterns.dll \
            Qt5Script.dll \
            Qt5Concurrent.dll \
            Qt5PrintSupport.dll \
            Qt5Widgets.dll \
            Qt5Multimedia.dll \
            Qt5MultimediaWidgets.dll \
            Qt5Quick.dll \
            Qt5Qml.dll \
            icuin51.dll \
            icudt51.dll \
            icuuc51.dll \
            libwinpthread-1.dll \
	    libgcc_s_dw2-1.dll \
	    libstdc++-6.dll \
            libwinpthread-1.dll \
            Qt5SerialPort.dll

        for(dll, QT_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_BINS]/$$dll\") $$targetPath(\"$$GCS_APP_PATH/$$dll\") $$addNewline()
        }

        message($$MINGW_PATH)

        # copy iconengines
        QT_ICONENGINE_DLLS = qsvgicon.dll
        data_copy.commands += -@$(MKDIR) $$targetPath(\"$$GCS_APP_PATH/iconengines\") $$addNewline()
        for(dll, QT_ICONENGINE_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_PLUGINS]/iconengines/$$dll\") $$targetPath(\"$$GCS_APP_PATH/iconengines/$$dll\") $$addNewline()
        }

        # copy imageformats
        QT_IMAGEFORMAT_DLLS = qgif.dll qico.dll qjpeg.dll qmng.dll qsvg.dll qtiff.dll
        data_copy.commands += -@$(MKDIR) $$targetPath(\"$$GCS_APP_PATH/imageformats\") $$addNewline()
        for(dll, QT_IMAGEFORMAT_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_PLUGINS]/imageformats/$$dll\") $$targetPath(\"$$GCS_APP_PATH/imageformats/$$dll\") $$addNewline()
        }

        # copy platforms
        QT_PLATFORMS_DLLS = qwindows.dll
        data_copy.commands += -@$(MKDIR) $$targetPath(\"$$GCS_APP_PATH/platforms\") $$addNewline()
        for(dll, QT_PLATFORMS_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_PLUGINS]/platforms/$$dll\") $$targetPath(\"$$GCS_APP_PATH/platforms/$$dll\") $$addNewline()
        }

        # copy phonon_backend
        #QT_PHONON_BACKEND_DLLS = phonon_ds94.dll
        #data_copy.commands += -@$(MKDIR) $$targetPath(\"$$GCS_APP_PATH/phonon_backend\") $$addNewline()
        #for(dll, QT_PHONON_BACKEND_DLLS) {
        #    data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_PLUGINS]/phonon_backend/$$dll\") $$targetPath(\"$$GCS_APP_PATH/phonon_backend/$$dll\") $$addNewline()
        #}

        # copy sqldrivers
        QT_SQLDRIVERS_DLLS = qsqlite.dll
        data_copy.commands += -@$(MKDIR) $$targetPath(\"$$GCS_APP_PATH/sqldrivers\") $$addNewline()
        for(dll, QT_SQLDRIVERS_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$[QT_INSTALL_PLUGINS]/sqldrivers/$$dll\") $$targetPath(\"$$GCS_APP_PATH/sqldrivers/$$dll\") $$addNewline()
        }

        SDL {
            # copy SDL - Simple DirectMedia Layer (www.libsdl.org)
            # Check the wiki for SDL installation, it should be copied first
            # (make sure that the Qt installation path below is correct)
            #
            # - For qt-sdk-win-opensource-2010.05.exe:
            #   xcopy /s /e <SDL>\bin\SDL.dll   C:\Qt\2010.05\mingw\bin\SDL.dll
            #   xcopy /s /e <SDL>\include\SDL\* C:\Qt\2010.05\mingw\include\SDL
            #   xcopy /s /e <SDL>\lib\*         C:\Qt\2010.05\mingw\lib
            #
            # - For Qt_SDK_Win_offline_v1_1_1_en.exe:
            #   xcopy /s /e <SDL>\bin\SDL.dll   C:\QtSDK\Desktop\Qt\4.7.3\mingw\bin\SDL.dll
            #   xcopy /s /e <SDL>\include\SDL\* C:\QtSDK\Desktop\Qt\4.7.3\mingw\include\SDL
            #   xcopy /s /e <SDL>\lib\*         C:\QtSDK\Desktop\Qt\4.7.3\mingw\lib
            SDL_DLL = SDL.dll
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$(QTMINGW)$$SDL_DLL\") $$targetPath(\"$$GCS_APP_PATH/$$SDL_DLL\") $$addNewline()
        }
        data_copy.target = FORCE
        QMAKE_EXTRA_TARGETS += data_copy
    }

    # copy OpenSSL DLLs
    {
        THIRDPARTY_PATH = $$GCS_SOURCE_TREE/../../tools
        OPENSSL_DIR = $$THIRDPARTY_PATH/win32openssl
        win32 {
        OPENSSL_DLLS = \
            ssleay32.dll \
            libeay32.dll
        }
        for(dll, OPENSSL_DLLS) {
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$OPENSSL_DIR/$$dll\") $$targetPath(\"$$GCS_APP_PATH/$$dll\") $$addNewline()
        }
        data_copy.target = FORCE
        QMAKE_EXTRA_TARGETS += data_copy
    }

    # Copy KML libraries
    KML {
        THIRDPARTY_PATH = $$GCS_SOURCE_TREE/../../tools
        linux-g++* {
            # Copy extra binary library files
            EXTRA_BINFILES += \
                $${THIRDPARTY_PATH}/libkml/lib/libkmlbase.so.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libkmlbase.so.0.0.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libkmldom.so.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libkmldom.so.0.0.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libkmlengine.so.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libkmlengine.so.0.0.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libminizip.so.0 \
                $${THIRDPARTY_PATH}/libkml/lib/libminizip.so.0.0.0 \
                $${THIRDPARTY_PATH}/libkml/lib/liburiparser.so.1 \
                $${THIRDPARTY_PATH}/libkml/lib/liburiparser.so.1.0.5
        }

        for(FILE,EXTRA_BINFILES){
            data_copy.commands += $(COPY_FILE) $$targetPath(\"$$FILE\") $$targetPath(\"$$GCS_LIBRARY_PATH\") $$addNewline()
        }
        data_copy.target = FORCE
        QMAKE_EXTRA_TARGETS += data_copy
    }
}
