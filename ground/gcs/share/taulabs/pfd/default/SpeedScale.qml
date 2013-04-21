import Qt 4.7

Item {
    id: sceneItem
    property variant sceneSize
    property real calibratedAirspeed : 3.6 * AirspeedActual.CalibratedAirspeed


    // Create speed ticker
    SvgElementImage {
        id: speed_bg
        elementName: "speed-bg"
        sceneSize: sceneItem.sceneSize
        clip: true

        x: Math.floor(scaledBounds.x * sceneItem.width)
        y: Math.floor(scaledBounds.y * sceneItem.height)

        // Create speed ticker
        SvgElementImage {
            id: speed_scale

            elementName: "speed-scale"
            sceneSize: sceneItem.sceneSize

            anchors.verticalCenter: parent.verticalCenter
            // The speed scale shows 30 kph of range
            // move it in 0..5m/s increments
            anchors.verticalCenterOffset: unitHeight * (sceneItem.calibratedAirspeed-Math.floor(sceneItem.calibratedAirspeed/5)*5)
            anchors.right: parent.right

            property int topNumber: Math.floor(sceneItem.calibratedAirspeed/5)*5+15
            property int bottomNumber: Math.floor(sceneItem.calibratedAirspeed/5)*5-15
            property real unitHeight: speed_scale.height / 30

            SvgElementImage {
                id: speed_desired

                elementName: "speed-desired"
                sceneSize: sceneItem.sceneSize

                // TODO: Update this to show desired calibrated airspeed
                property real desiredSpeed : 3.6 * PathDesired.EndingVelocity

                anchors.right: parent.right
                anchors.verticalCenter: parent.top
                anchors.verticalCenterOffset: (speed_scale.topNumber-desiredSpeed)*speed_scale.unitHeight
            }

            // speed numbers
            Column {
                width: speed_bg.width
                anchors.right: speed_scale.right

                Repeater {
                    model: 7
                    Item {
                        height: speed_scale.height / 6
                        width: speed_bg.width

                        Text {
                            //don't show negative numbers
                            text: speed_scale.topNumber - index*5
                            color: "white"
                            visible: speed_scale.topNumber - index*5 >= 0

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
        id: speed_desired_offscale

        elementName: "setpoint-bug-offscale"
        sceneSize: sceneItem.sceneSize

        property int topVisibleNumber: sceneItem.calibratedAirspeed + 13
        property int bottomVisibleNumber: sceneItem.calibratedAirspeed - 13

        rotation: ((topVisibleNumber-speed_desired.desiredSpeed) > 0) * 180
        visible: (topVisibleNumber-speed_desired.desiredSpeed) < 0 || (bottomVisibleNumber-speed_desired.desiredSpeed) > 0

        anchors.right: speed_bg.right
        anchors.verticalCenter: ((topVisibleNumber-speed_desired.desiredSpeed) < 0 ? speed_bg.top : speed_bg.bottom)
        anchors.verticalCenterOffset:  ((topVisibleNumber-speed_desired.desiredSpeed) < 0 ? -speed_desired_offscale.height/2.0-sceneItem.height*.003 : speed_desired_offscale.height/2.0+sceneItem.height*.003)
    }

    // Add text to speed ticker
    SvgElementImage {
        id: speed_window
        clip: true

        elementName: "speed-window"
        sceneSize: sceneItem.sceneSize
        anchors.centerIn: speed_bg

        Text {
            id: speed_text
            text: Math.round(sceneItem.calibratedAirspeed).toFixed()
            color: "white"
            font {
                family: "Arial"
                pixelSize: parent.height * 0.6
            }
            anchors.centerIn: parent
        }
    }
}
