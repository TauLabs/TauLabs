#
# Project: TauLabs
# NSIS configuration file for TauLabs GCS
# Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
# The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2010-2012.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#

# This script requires Unicode NSIS 2.46 or higher:
# http://www.scratchpaper.com/

# TODO:
#  - install only built/used modules, not a whole directory.
#  - remove only installed files, not a whole directory.

;--------------------------------
; Includes

!include "x64.nsh"

;--------------------------------
; Paths

  ; Tree root locations (relative to this script location)
  !define PROJECT_ROOT   "..\.."
  !define NSIS_DATA_TREE "."
  !define GCS_BUILD_TREE "..\..\build\ground\gcs"
  !define UAVO_SYNTH_TREE "..\..\build\uavobject-synthetics"
  !define MATLAB_TREE "..\..\build\matlab"
  !define AEROSIMRC_TREE "..\..\build\ground\AeroSIM-RC"

  ; Default installation folder
  InstallDir "$PROGRAMFILES\TauLabs"

  ; Get installation folder from registry if available
  InstallDirRegKey HKLM "Software\TauLabs" "Install Location"

;--------------------------------
; Version information

  ; Program name and installer file
  !define PRODUCT_NAME "TauLabs GCS"
  !define INSTALLER_NAME "TauLabs GCS Installer"

  ; Read automatically generated version info
; !define PACKAGE_LBL "${DATE}-${TAG_OR_HASH8}"
; !define PACKAGE_DIR "..\..\build\package-$${PACKAGE_LBL}"
; !define OUT_FILE "TauLabs-$${PACKAGE_LBL}-install.exe"
; !define FIRMWARE_DIR "firmware-$${PACKAGE_LBL}"
; !define PRODUCT_VERSION "0.0.0.0"
; !define FILE_VERSION "${TAG_OR_BRANCH}:${HASH8} ${DATETIME}"
; !define BUILD_DESCRIPTION "${TAG_OR_BRANCH}:${HASH8} built using ${ORIGIN} as origin, committed ${DATETIME} as ${HASH}"
  !include "${GCS_BUILD_TREE}\taulabs.nsh"

  Name "${PRODUCT_NAME}"
  OutFile "${PACKAGE_DIR}\${OUT_FILE}"

  VIProductVersion ${PRODUCT_VERSION}
  VIAddVersionKey "ProductName" "${INSTALLER_NAME}"
  VIAddVersionKey "FileVersion" "${FILE_VERSION}"
  VIAddVersionKey "Comments" "${INSTALLER_NAME}. ${BUILD_DESCRIPTION}"
  VIAddVersionKey "CompanyName" "TauLabs, http://taulabs.org"
  VIAddVersionKey "LegalCopyright" "© 2012-2013 Tau Labs, 2010-2012 The OpenPilot Team"
  VIAddVersionKey "FileDescription" "${INSTALLER_NAME}"

;--------------------------------
; Installer interface and base settings

  !include "MUI2.nsh"
  !define MUI_ABORTWARNING

  ; Adds an XP manifest to the installer
  XPStyle on

  ; Request application privileges for Windows Vista/7
  RequestExecutionLevel admin

  ; Compression level
  SetCompressor /solid lzma

;--------------------------------
; Branding

  BrandingText "© 2012-2013 Tau Labs, http://taulabs.org. 2010-2012 The OpenPilot Team, http://www.openpilot.org"

  !define MUI_ICON "${NSIS_DATA_TREE}\resources\taulabs.ico"
  !define MUI_HEADERIMAGE
  !define MUI_HEADERIMAGE_BITMAP "${NSIS_DATA_TREE}\resources\header.bmp"
  !define MUI_HEADERIMAGE_BITMAP_NOSTRETCH
  !define MUI_WELCOMEFINISHPAGE_BITMAP "${NSIS_DATA_TREE}\resources\welcome.bmp"
  !define MUI_WELCOMEFINISHPAGE_BITMAP_NOSTRETCH
  !define MUI_UNWELCOMEFINISHPAGE_BITMAP "${NSIS_DATA_TREE}\resources\welcome.bmp"
  !define MUI_UNWELCOMEFINISHPAGE_BITMAP_NOSTRETCH

;--------------------------------
; Language selection dialog settings

  ; Remember the installer language
  !define MUI_LANGDLL_REGISTRY_ROOT "HKCU" 
  !define MUI_LANGDLL_REGISTRY_KEY "Software\TauLabs" 
  !define MUI_LANGDLL_REGISTRY_VALUENAME "Installer Language"
  !define MUI_LANGDLL_ALWAYSSHOW

;--------------------------------
; Settings for MUI_PAGE_FINISH
  !define MUI_FINISHPAGE_RUN
  !define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\HISTORY.txt"
  !define MUI_FINISHPAGE_RUN_FUNCTION "RunApplication"

;--------------------------------
; Pages

  !insertmacro MUI_PAGE_WELCOME
  !insertmacro MUI_PAGE_LICENSE "$(LicenseFile)"
  !insertmacro MUI_PAGE_COMPONENTS
  !insertmacro MUI_PAGE_DIRECTORY
  !insertmacro MUI_PAGE_INSTFILES
  !insertmacro MUI_PAGE_FINISH

  !insertmacro MUI_UNPAGE_WELCOME
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_COMPONENTS
  !insertmacro MUI_UNPAGE_INSTFILES
  !insertmacro MUI_UNPAGE_FINISH

;--------------------------------
; Supported languages, license files and translations

  !include "${NSIS_DATA_TREE}\translations\languages.nsh"

;--------------------------------
; Reserve files

  ; If you are using solid compression, files that are required before
  ; the actual installation should be stored first in the data block,
  ; because this will make your installer start faster.

  !insertmacro MUI_RESERVEFILE_LANGDLL

;--------------------------------
; Installer sections

; Copy GCS core files
Section "Core files" InSecCore
  SectionIn RO
  SetOutPath "$INSTDIR\bin"
  File /r "${GCS_BUILD_TREE}\bin\*"
  SetOutPath "$INSTDIR"
  File "${PROJECT_ROOT}\HISTORY.txt"
SectionEnd

; Copy GCS plugins
Section "-Plugins" InSecPlugins
  SectionIn RO
  SetOutPath "$INSTDIR\lib\taulabs\plugins"
  File /r "${GCS_BUILD_TREE}\lib\taulabs\plugins\*.dll"
  File /r "${GCS_BUILD_TREE}\lib\taulabs\plugins\*.pluginspec"
SectionEnd

; Copy GCS resources
Section "-Resources" InSecResources
  SetOutPath "$INSTDIR\share\taulabs\default_configurations"
  File /r "${GCS_BUILD_TREE}\share\taulabs\default_configurations\*"
  SetOutPath "$INSTDIR\share\taulabs\stylesheets"
  File /r "${GCS_BUILD_TREE}\share\taulabs\stylesheets\*"
  SetOutPath "$INSTDIR\share\taulabs\diagrams"
  File /r "${GCS_BUILD_TREE}\share\taulabs\diagrams\*"
  SetOutPath "$INSTDIR\share\taulabs\dials"
  File /r "${GCS_BUILD_TREE}\share\taulabs\dials\*"
  SetOutPath "$INSTDIR\share\taulabs\mapicons"
  File /r "${GCS_BUILD_TREE}\share\taulabs\mapicons\*"
  SetOutPath "$INSTDIR\share\taulabs\models"
  File /r "${GCS_BUILD_TREE}\share\taulabs\models\*"
  SetOutPath "$INSTDIR\share\taulabs\pfd"
  File /r "${GCS_BUILD_TREE}\share\taulabs\pfd\*"
SectionEnd

; Copy Notify plugin sound files
Section "-Sound files" InSecSounds
  SetOutPath "$INSTDIR\share\taulabs\sounds"
  File /r "${GCS_BUILD_TREE}\share\taulabs\sounds\*"
SectionEnd

; Copy localization files
; Disabled until GCS source is stable and properly localized
Section "-Localization" InSecLocalization
  SetOutPath "$INSTDIR\share\taulabs\translations"
; File /r "${GCS_BUILD_TREE}\share\taulabs\translations\taulabs_*.qm"
  File /r "${GCS_BUILD_TREE}\share\taulabs\translations\qt_*.qm"
SectionEnd

; Copy firmware files
Section "Firmware" InSecFirmware
; SetOutPath "$INSTDIR\firmware\${FIRMWARE_DIR}"
; File /r "${PACKAGE_DIR}\${FIRMWARE_DIR}\*"
  SetOutPath "$INSTDIR\firmware"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_coptercontrol-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_pipxtreme-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_flyingf3-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_flyingf4-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_quanton-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_colibri-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_freedom-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_sparky-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_revolution-${PACKAGE_LBL}.tlfw"
  File "${PACKAGE_DIR}\${FIRMWARE_DIR}\fw_revomini-${PACKAGE_LBL}.tlfw"
SectionEnd

; Copy utility files
Section "-Utilities" InSecUtilities
  SetOutPath "$INSTDIR\utilities"
  File "/oname=LogConvert-${PACKAGE_LBL}.m" "${MATLAB_TREE}\LogConvert.m"
SectionEnd

; Copy driver files
Section "-Drivers" InSecDrivers
  SetOutPath "$INSTDIR\drivers"
  File "${PROJECT_ROOT}\flight\Project\Windows USB\TauLabs-CDC.inf"
SectionEnd

; Preinstall OpenPilot CDC driver
Section "CDC driver" InSecInstallDrivers
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  ${If} ${RunningX64}
    File "/oname=dpinst.exe" "${NSIS_DATA_TREE}\redist\dpinst_x64.exe"
  ${Else}
    File "/oname=dpinst.exe" "${NSIS_DATA_TREE}\redist\dpinst_x86.exe"
  ${EndIf}
  ExecWait '"$PLUGINSDIR\dpinst.exe" /lm /path "$INSTDIR\drivers"'
SectionEnd

; AeroSimRC plugin files
Section "AeroSimRC plugin" InSecAeroSimRC
  SetOutPath "$INSTDIR\misc\AeroSIM-RC"
  File /r "${AEROSIMRC_TREE}\*"
SectionEnd

Section "Shortcuts" InSecShortcuts
  ; Create desktop and start menu shortcuts
  SetOutPath "$INSTDIR"
  CreateDirectory "$SMPROGRAMS\Tau Labs"
  CreateShortCut "$SMPROGRAMS\Tau Labs\Tau Labs GCS.lnk" "$INSTDIR\bin\taulabsgcs.exe" \
	"" "$INSTDIR\bin\taulabsgcs.exe" 0 "" "" "${PRODUCT_NAME} ${PRODUCT_VERSION}. ${BUILD_DESCRIPTION}"
  CreateShortCut "$SMPROGRAMS\Tau Labs\Tau Labs GCS (clean configuration).lnk" "$INSTDIR\bin\taulabsgcs.exe" \
	"-clean-config" "$INSTDIR\bin\taulabsgcs.exe" 0 "" "" "${PRODUCT_NAME} ${PRODUCT_VERSION}. ${BUILD_DESCRIPTION}"
  CreateShortCut "$SMPROGRAMS\Tau Labs\Tau Labs ChangeLog.lnk" "$INSTDIR\HISTORY.txt" \
	"" "$INSTDIR\bin\taulabsgcs.exe" 0
  CreateShortCut "$SMPROGRAMS\Tau Labs\Tau Labs Website.lnk" "http://taulabs.org" \
	"" "$INSTDIR\bin\taulabsgcs.exe" 0
  CreateShortCut "$DESKTOP\Tau Labs GCS.lnk" "$INSTDIR\bin\taulabsgcs.exe" \
  	"" "$INSTDIR\bin\taulabsgcs.exe" 0 "" "" "${PRODUCT_NAME} ${PRODUCT_VERSION}. ${BUILD_DESCRIPTION}"
  CreateShortCut "$SMPROGRAMS\Tau Labs\Uninstall.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0
SectionEnd

Section ; create uninstall info
  ; Write the installation path into the registry
  WriteRegStr HKCU "Software\TauLabs" "Install Location" $INSTDIR

  ; Write the uninstall keys for Windows
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TauLabs" "DisplayName" "Tau Labs GCS"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TauLabs" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TauLabs" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TauLabs" "NoRepair" 1

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

;--------------------------------
; Installer section descriptions

  !insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecCore} $(DESC_InSecCore)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecPlugins} $(DESC_InSecPlugins)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecResources} $(DESC_InSecResources)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecSounds} $(DESC_InSecSounds)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecLocalization} $(DESC_InSecLocalization)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecFirmware} $(DESC_InSecFirmware)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecUtilities} $(DESC_InSecUtilities)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecDrivers} $(DESC_InSecDrivers)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecInstallDrivers} $(DESC_InSecInstallDrivers)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecAeroSimRC} $(DESC_InSecAeroSimRC)
    !insertmacro MUI_DESCRIPTION_TEXT ${InSecShortcuts} $(DESC_InSecShortcuts)
  !insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Installer functions

Function .onInit

  SetShellVarContext all
  !insertmacro MUI_LANGDLL_DISPLAY

FunctionEnd

;--------------------------------
; Uninstaller sections

Section "un.Tau Labs GCS" UnSecProgram
  ; Remove installed files and/or directories
  RMDir /r /rebootok "$INSTDIR\bin"
  RMDir /r /rebootok "$INSTDIR\lib"
  RMDir /r /rebootok "$INSTDIR\share"
  RMDir /r /rebootok "$INSTDIR\firmware"
  RMDir /r /rebootok "$INSTDIR\utilities"
  RMDir /r /rebootok "$INSTDIR\drivers"
  RMDir /r /rebootok "$INSTDIR\misc"
  Delete /rebootok "$INSTDIR\HISTORY.txt"
  Delete /rebootok "$INSTDIR\Uninstall.exe"

  ; Remove directory
  RMDir /rebootok "$INSTDIR"

  ; Remove registry keys
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TauLabs"
  DeleteRegKey HKCU "Software\TauLabs"

  ; Remove shortcuts, if any
  SetShellVarContext all
  Delete /rebootok "$DESKTOP\Tau Labs GCS.lnk"
  Delete /rebootok "$SMPROGRAMS\Tau Labs\*"
  RMDir /rebootok "$SMPROGRAMS\Tau Labs"
SectionEnd

Section "un.Maps cache" UnSecCache
  ; Remove maps cache
  SetShellVarContext current
  RMDir /r /rebootok "$APPDATA\TauLabs\mapscache"
SectionEnd

Section /o "un.Configuration" UnSecConfig
  ; Remove configuration
  SetShellVarContext current
  Delete /rebootok "$APPDATA\TauLabs\TauLabs*.db"
  Delete /rebootok "$APPDATA\TauLabs\TauLabs*.xml"
  Delete /rebootok "$APPDATA\TauLabs\TauLabs*.ini"
SectionEnd

Section "-un.Profile" UnSecProfile
  ; Remove TauLabs user profile subdirectory if empty
  SetShellVarContext current
  RMDir "$APPDATA\TauLabs"
SectionEnd

;--------------------------------
; Uninstall section descriptions

  !insertmacro MUI_UNFUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${UnSecProgram} $(DESC_UnSecProgram)
    !insertmacro MUI_DESCRIPTION_TEXT ${UnSecCache} $(DESC_UnSecCache)
    !insertmacro MUI_DESCRIPTION_TEXT ${UnSecConfig} $(DESC_UnSecConfig)
  !insertmacro MUI_UNFUNCTION_DESCRIPTION_END

;--------------------------------
; Uninstaller functions

Function un.onInit

  SetShellVarContext all
  !insertmacro MUI_UNGETLANGUAGE

FunctionEnd

;--------------------------------
; Function to run the application from installer

Function RunApplication

  Exec '"$INSTDIR\bin\taulabsgcs.exe"'

FunctionEnd
