// import QtQuick 1.0 // to target S60 5th Edition or Maemo 5
import QtQuick 1.1

Item {
    id: container
    width: 100
    height: 62

    signal clicked(string url)

    Text {
        id: header
        text: "Above Ground Labs Links"
        width: parent.width
        color: "#44515c"
        font {
            pointSize: 14
            weight: Font.Bold
        }
    }

    ListModel {
        id: sitesModel
        ListElement { title: "Above Ground Labs Home"; link: "http://abovegroundlabs.org" }
        ListElement { title: "Above Ground Labs Wiki"; link: "https://github.com/PhoenixPilot/PhoenixPilot/wiki" }
        ListElement { title: "Above Ground Labs Forums"; link: "https://groups.google.com/forum/#!forum/phoenixpilot" }
        ListElement { title: "Above Ground Labs Code"; link: "https://github.com/PhoenixPilot/PhoenixPilot" }
        ListElement { title: "Above Ground Labs Issues"; link: "https://github.com/PhoenixPilot/PhoenixPilot/issues" }
    }

    ListView {
        id: view
        width: parent.width
        anchors { top: header.bottom; topMargin: 14; bottom: parent.bottom }
        model: sitesModel
        spacing: 8
        clip: true

        delegate: Text {
            text: title
            width: view.width
            wrapMode: Text.WrapAtWordBoundaryOrAnywhere

            font {
                pointSize: 12
                weight: Font.Bold
            }

            color: mouseArea.containsMouse ? "#224d81" : "black"

            MouseArea {
                id: mouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    console.log(link)
                    container.clicked(link)
                }
            }
        }
    }
}
