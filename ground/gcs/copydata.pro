include(gcs.pri)

TEMPLATE = subdirs
# Copy Qt runtime libraries into the build directory (to run or package)
equals(copydata, 1) {
    # Windows release only, no debug target DLLs ending with 'd'
    win32:CONFIG(release, debug|release) {

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
