import QtQuick 1.1

Rectangle {
    id: container
    width: 1024
    height: 768
    gradient: Gradient {
        GradientStop {
            position: 0
            color: "#333333"
        }

        GradientStop {
            position: 1
            color: "#232323"
        }
    }

    Column {
        id: buttonsGrid
        anchors.horizontalCenter: parent.horizontalCenter
        // distribute a vertical space between the icons blocks an community widget as:
        // top - 48% - Icons - 27% - CommunityWidget - 25% - bottom
        y: (parent.height - buttons.height - communityPanel.height) * 0.48
        width: parent.width
        spacing: (parent.height - buttons.height - communityPanel.height) * 0.27

        Row {
            //if the buttons grid overlaps vertically with the wizard buttons,
            //move it left to use only the space left to wizard buttons
            property real availableWidth: container.width
            x: (availableWidth-width)/2
            spacing: 16

            Image {
                x: -56
                sourceSize.height: 235
                sourceSize.width: 234
                source: "images/welcome-logo.png"
                anchors.verticalCenter: parent.verticalCenter
                anchors.verticalCenterOffset: -2 //it looks better aligned to icons grid

                //hide the logo on the very small screen to fit the buttons
                visible: parent.availableWidth > width + parent.spacing + buttons.width + wizard.width
            }

            Grid {
                id: buttons
                columns: 3
                spacing: 4
                anchors.verticalCenter: parent.verticalCenter

                WelcomePageButton {
                    baseIconName: "flightdata"
                    label: "Flight Data"
                    onClicked: welcomePlugin.openPage("Flight data")
                }

                WelcomePageButton {
                    baseIconName: "config"
                    label: "Configuration"
                    onClicked: welcomePlugin.openPage("Configuration")
                }

                WelcomePageButton {
                    baseIconName: "system"
                    label: "System"
                    onClicked: welcomePlugin.openPage("System")
                }

               WelcomePageButton {
                    baseIconName: "scopes"
                    label: "Scopes"
                    onClicked: welcomePlugin.openPage("Scopes")
                }

                WelcomePageButton {
                    baseIconName: "hitl"
                    label: "HITL"
                    onClicked: welcomePlugin.openPage("HITL")
                }

                WelcomePageButton {
                    baseIconName: "firmware"
                    label: "Firmware"
                    onClicked: welcomePlugin.openPage("Firmware")
                }
            } //icons grid

            WelcomePageButton {
                id: wizard
                anchors.verticalCenter: parent.verticalCenter
                baseIconName: "wizard"
                onClicked: welcomePlugin.triggerAction("SetupWizardPlugin.ShowSetupWizard")
            }

        } // images row

        CommunityPanel {
            id: communityPanel
            anchors.horizontalCenter: parent.horizontalCenter
            width: Math.min(sourceSize.width, container.width)
            height: Math.min(300, container.height*0.5)
        }
    }
}
