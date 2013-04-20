git clone git://github.com/TauLabs/TauLabs.git TauLabs
cd TauLabs
mkdir downloads
curl -o qt-windows-opensource-5.0.2-mingw47_32-x86-offline.exe http://origin.releases.qt-project.org/qt5/5.0.2/qt-windows-opensource-5.0.2-mingw47_32-x86-offline.exe
echo "Please install into `pwd`/tools/Qt5.0.2"
./downloads/qt-windows-opensource-5.0.2-mingw47_32-x86-offline.exe
export PATH=${PATH}:`pwd`/tools/Qt5.0.2/Tools/MinGW/bin/
