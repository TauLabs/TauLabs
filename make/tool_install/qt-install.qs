/*
silent installer script

known not work with Qt 5.5.1 and QtIFW 2.0.2

known issues:
- silent but not headless (QtIFW 2.0.3 should support installer.setSilent(true))
- cannot disable forced components (QtCreator, ...)
 - cannot disable virtual components (doc, examples, ...)
 - cannot disable shortcuts creation
 - if user presses the 'Show Details' button then the installer does not end automatically
*/
var qtMajorVersion;
var qtMinorVersion;

function Controller()
{
    console.log("*** Silent Installer ***");
    console.log("Installing on " + installer.value("os"));
    //installer.setSilent(true);

    var qtInstallTargetDir = installer.environmentVariable("QT_INSTALL_TARGET_DIR");
    if (qtInstallTargetDir == "") {
        qtInstallTargetDir = installer.environmentVariable("PWD") + "/tools/qt-" + qtMajorVersion + "." + qtMinorVersion + "." + qtPatchVersion;
        console.log("Environment variable QT_INSTALL_TARGET_DIR not set, using default " + qtInstallTargetDir);
    }

    // Get the major and minor versions from the environment
    qtMajorVersion = installer.environmentVariable("QT_MAJOR_VERSION");
    qtMinorVersion = installer.environmentVariable("QT_MINOR_VERSION");

    installer.setValue("TargetDir", qtInstallTargetDir);
    console.log("Installing to " + installer.value("TargetDir"));

    installer.autoRejectMessageBoxes();
    installer.setMessageBoxAutomaticAnswer("OverwriteTargetDirectory", QMessageBox.Yes);
    installer.setMessageBoxAutomaticAnswer("stopProcessesForUpdates", QMessageBox.Ignore);

    // pages that are not visible are actually removed from the wizard
    // some pages must not be removed otherwise the installer starts to mishbehave
    installer.setDefaultPageVisible(QInstaller.Welcome, false);
    installer.setDefaultPageVisible(QInstaller.Credentials, false); // QInstaller.Credentials is 0... so this is a NOP!
    //installer.setDefaultPageVisible(QInstaller.Introduction, false); // Fails to skip Credentials if Introduction is removed?
    installer.setDefaultPageVisible(QInstaller.TargetDirectory, false);
    //installer.setDefaultPageVisible(QInstaller.ComponentSelection, false);
    //installer.setDefaultPageVisible(QInstaller.LicenseAgreementCheck, false);
    //installer.setDefaultPageVisible(QInstaller.StartMenuSelection, false);
    installer.setDefaultPageVisible(QInstaller.ReadyForInstallation, false);
    //installer.setDefaultPageVisible(QInstaller.PerformInstallation, false);
    installer.setDefaultPageVisible(QInstaller.FinishedPage, false);

    installer.componentAdded.connect(onComponentAdded);
    installer.aboutCalculateComponentsToInstall.connect(onAboutCalculateComponentsToInstall);
    installer.finishedCalculateComponentsToInstall.connect(onFinishedCalculateComponentsToInstall);
}

// installer callbacks

onComponentAdded = function(component)
{
    console.log("Component added: " + component.name);
    dumpComponents();
}

onAboutCalculateComponentsToInstall = function()
{
    console.log("onAboutCalculateComponentsToInstall");
    //dumpComponents();
}

onFinishedCalculateComponentsToInstall = function()
{
    console.log("onFinishedCalculateComponentsToInstall");
    //dumpComponents();
}

// page callbacks
// used to setup wizard pages and move the wizard forward

Controller.prototype.WelcomePageCallback = function()
{
    logCallback();

    gui.clickButton(buttons.NextButton);
}

Controller.prototype.CredentialsPageCallback = function()
{
    logCallback();

    gui.clickButton(buttons.NextButton);
}

Controller.prototype.IntroductionPageCallback = function()
{
    logCallback();

    gui.clickButton(buttons.NextButton);
}

Controller.prototype.ComponentSelectionPageCallback = function()
{
    logCallback();

    var page = gui.currentPageWidget();
    page.deselectAll()
    if (installer.value("os") == "win") {
        selectComponent(page, "qt." + qtMajorVersion + qtMinorVersion + ".win32_mingw492");
        selectComponent(page, "qt.tools.win32_mingw492");
    }
    else if (installer.value("os") == "x11") {
        selectComponent(page, "qt." + qtMajorVersion + qtMinorVersion + ".gcc");
        selectComponent(page, "qt." + qtMajorVersion + qtMinorVersion + ".gcc_64");
    }
    else if (installer.value("os") == "mac") {
        selectComponent(page, "qt." + qtMajorVersion + qtMinorVersion + ".clang_64");
    }
    selectComponent(page, "qt." + qtMajorVersion + qtMinorVersion + ".qtquickcontrols");
    selectComponent(page, "qt." + qtMajorVersion + qtMinorVersion + ".qtscript");

    //installer.componentByName("qt.tools.qtcreator").setValue("ForcedInstallation", "false");

    gui.clickButton(buttons.NextButton);
}


function selectComponent(page, name)
{
    component = installer.componentByName(name);
    if (component) {
        console.log("component " + name + " : " + component);
        page.selectComponent(name);
    }
    else  {
        console.log("Failed to find component " + name + "!");
    }
}

Controller.prototype.LicenseAgreementPageCallback = function()
{
    logCallback();

    setChecked("AcceptLicenseRadioButton", true);

    gui.clickButton(buttons.NextButton);
}

Controller.prototype.StartMenuDirectoryPageCallback = function()
{
    logCallback();

    gui.clickButton(buttons.NextButton);
}

Controller.prototype.PerformInstallationPageCallback = function()
{
    logCallback();

    // show details and hide button
    click("DetailsButton");
    setVisible("DetailsButton", false);

    // showing details will disable automated page switch, so re-enable it
    installer.setAutomatedPageSwitchEnabled(true);
}

Controller.prototype.FinishedPageCallback = function()
{
    logCallback();

    setChecked("launchQtCreatorCheckBox", false);

    gui.clickButton(buttons.FinishButton);
}

// utilities

function logCallback()
{
    var page = gui.currentPageWidget();
    console.log(">>> " + page.objectName + "Callback");
}

function dumpComponents()
{
    dumpComponentsArray(installer.components());
}

function dumpComponentsArray(components)
{
    var arrayLength = components.length;
    for (var i = 0; i < arrayLength; i++) {
        dumpComponent(components[i]);
    }
}

function dumpComponent(component)
{
    console.log(component.name + " (" + component.displayName + ")");
    console.log("  Virtual: " + component.value("Virtual", "false"));
    console.log("  ForcedInstallation: " + component.value("ForcedInstallation", "false"));
    console.log("  Default: " + component.default);
    console.log("  Enabled: " + component.enabled);
}

// UI utilities

function click(name)
{
    var page = gui.currentPageWidget();
    var button = gui.findChild(page, name);
    if (button) {
        console.log("button " + name + " : " + button);
        button.click();
    }
    else {
        console.log("Failed to find button " + name + "!");
    }
}

function setVisible(name, visible)
{
    var page = gui.currentPageWidget();
    var button = gui.findChild(page, name);
    if (button) {
        console.log("button " + name + " : " + button);
        button.visible = visible;
        console.log("button " + name + " visible : " + button.visible);
    }
    else {
        console.log("Failed to find button " + name + "!");
    }
}

function setEnabled(name, enabled)
{
    var page = gui.currentPageWidget();
    var button = gui.findChild(page, name);
    if (button) {
        console.log("button " + name + " : " + button);
        button.enabled = enabled;
        console.log("button " + name + " enabled : " + button.enabled);
    }
    else {
        console.log("Failed to find button " + name + "!");
    }
}

function setChecked(name, checked)
{
    var page = gui.currentPageWidget();
    var button = gui.findChild(page, name);
    if (button) {
        console.log("button " + name + " : " + button);
        button.checked = checked;
        console.log("button " + name + " checked : " + button.checked);
    }
    else {
        console.log("Failed to find button " + name + "!");
    }
}
