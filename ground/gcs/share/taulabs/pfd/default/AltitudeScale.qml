import Qt 4.7

Item {
    id: sceneItem
    property variant sceneSize

    SvgElementImage {
        id: altitude_bg
        elementName: "altitude-bg"
        sceneSize: sceneItem.sceneSize
        clip: true

        property variant scaledBounds: svgRenderer.scaledElementBounds("pfd.svg", "altitude-bg")

        x: Math.floor(scaledBounds.x * sceneItem.width)
        y: Math.floor(scaledBounds.y * sceneItem.height)

        SvgElementImage {
            id: altitude_scale

            elementName: "altitude-scale"
            sceneSize: sceneItem.sceneSize

            anchors.verticalCenter: parent.verticalCenter
            // The altitude scale represents 30 meters,
            // move it in 0..5m range
            anchors.verticalCenterOffset: unitHeight * (altitude - Math.floor(altitude/5)*5)
            anchors.left: parent.left

            property int topNumber: Math.floor(altitude/5)*5 + 15
            property int bottomNumber: Math.floor(altitude/5)*5 - 15
            property real unitHeight: altitude_scale.height / 30
            property real altitude: -PositionActual.Down

            SvgElementImage {
                id: altitude_desired

                elementName: "altitude-desired"
                sceneSize: sceneItem.sceneSize

                property real desiredAltitude : -PathDesired.End_Down

                anchors.left: parent.left
                anchors.verticalCenter: parent.top
                anchors.verticalCenterOffset: (altitude_scale.topNumber-desiredAltitude)*altitude_scale.unitHeight
            }

            // Altitude numbers
            Column {
                Repeater {
                    model: 7
                    Item {
                        height: altitude_scale.unitHeight * 5
                        width: altitude_bg.width

                        Text {
                            text: altitude_scale.topNumber - index*5
                            color: "white"
                            font.pixelSize: parent.height / 4
                            font.family: "Arial"

                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.verticalCenter: parent.top
                        }
                    }
                }
            }
        }
    }

    // Add off-scale chevrons
    SvgElementImage {
        id: altitude_desired_offscale

        elementName: "setpoint-bug-offscale"
        sceneSize: sceneItem.sceneSize

        property int topVisibleNumber: altitude_scale.altitude + 13
        property int bottomVisibleNumber: altitude_scale.altitude - 13

        rotation: ((topVisibleNumber-altitude_desired.desiredAltitude) > 0) * 180
        visible: (topVisibleNumber-altitude_desired.desiredAltitude) < 0 || (bottomVisibleNumber-altitude_desired.desiredAltitude) > 0

        anchors.left: altitude_bg.left
        anchors.verticalCenter: ((topVisibleNumber-altitude_desired.desiredAltitude) < 0 ? altitude_bg.top : altitude_bg.bottom)
        anchors.verticalCenterOffset:  ((topVisibleNumber-altitude_desired.desiredAltitude) < 0 ? -altitude_desired_offscale.height/2.0-sceneItem.height*.003 : altitude_desired_offscale.height/2.0+sceneItem.height*.003)
    }


    // Add text to speed ticker
    SvgElementImage {
        id: altitude_window
        clip: true

        elementName: "altitude-window"
        sceneSize: sceneItem.sceneSize
        anchors.centerIn: altitude_bg

        Text {
            id: altitude_text
            text: Math.floor(altitude_scale.altitude).toFixed()
            color: "white"
            font {
                family: "Arial"
                pixelSize: parent.height * 0.6
            }
            anchors.centerIn: parent
        }
    }
}
