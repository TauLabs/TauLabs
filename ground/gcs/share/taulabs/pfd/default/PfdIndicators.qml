import QtQuick 2.0

Item {
    id: sceneItem
    property variant sceneSize

    //AttitudeActual.Yaw is converted to -180..180 range
    property real yaw: sceneItem.parent.circular_modulus_deg(AttitudeActual.Yaw)
    property real pitch : (AttitudeActual.Pitch)

    // Telemetry status arrow
    SvgElementImage {
        id: telemetry_status
        elementName: "gcstelemetry-"+statusName
        sceneSize: sceneItem.sceneSize

        property string statusName : ["Disconnected","HandshakeReq","HandshakeAck","Connected"][GCSTelemetryStats.Status]

        // Force refresh of the arrow image when elementName changes
        onElementNameChanged: { generateSource() }

        scaledBounds: svgRenderer.scaledElementBounds("pfd.svg", "gcstelemetry-Disconnected")
        x: Math.floor(scaledBounds.x * sceneItem.width)
        y: Math.floor(scaledBounds.y * sceneItem.height)
    }

    // Telemetry rate text
    Text {
        id: telemetry_rate
        text: GCSTelemetryStats.TxDataRate.toFixed()+"/"+GCSTelemetryStats.RxDataRate.toFixed()
        color: "white"
        font.family: "Arial"
        font.pixelSize: telemetry_status.height * 0.75

        anchors.top: telemetry_status.bottom
        anchors.horizontalCenter: telemetry_status.horizontalCenter
    }

    // GPS status text
    Text {
        id: gps_text
        text: "GPS: " + GPSPosition.Satellites + "\nPDP: " + GPSPosition.PDOP.toFixed(2)
        color: "white"
        font.family: "Arial"
        font.pixelSize: telemetry_status.height * 0.5

        visible: GPSPosition.Satellites > 0

        property variant scaledBounds: svgRenderer.scaledElementBounds("pfd.svg", "gps-txt")
        x: Math.floor(scaledBounds.x * sceneItem.width)
        y: Math.floor(scaledBounds.y * sceneItem.height)
    }

    // Battery text
    Text {
        id: battery_text

        text: FlightBatteryState.Voltage.toFixed(2)+" V\n" +
              FlightBatteryState.Current.toFixed(2)+" A\n" +
              FlightBatteryState.ConsumedEnergy.toFixed()+" mAh"

        color: "white"
        font.family: "Arial"

        //I think it should be pixel size,
        //but making it more consistent with C++ version instead
        font.pointSize: scaledBounds.height * sceneItem.height

        visible: FlightBatteryState.Voltage > 0 || FlightBatteryState.Current > 0

        property variant scaledBounds: svgRenderer.scaledElementBounds("pfd.svg", "battery-txt")
        x: Math.floor(scaledBounds.x * sceneItem.width)
        y: Math.floor(scaledBounds.y * sceneItem.height)
    }

    // Draw home location marker
    Item {
        id: homelocation
        anchors.fill: parent

        transform: [
            Translate {
                id: homeLocationTranslate
                x: homelocation.parent.width/2*Math.sin(homewaypoint.bearing_R)*(Math.sin(Math.PI/2)/Math.sin(parent.fovX_D*Math.PI/180/2))
                y: -homelocation.parent.height/2*Math.sin(homewaypoint.elevation_R)*(Math.sin(Math.PI/2)/Math.sin(parent.fovY_D*Math.PI/180/2))
            },
            Rotation {
                angle: -AttitudeActual.Roll
                origin.x : homelocation.parent.width/2
                origin.y : homelocation.parent.height/2
            }
        ]

        SvgElementImage {
            id: homewaypoint

            elementName: "homewaypoint"
            sceneSize: sceneItem.sceneSize

            // Home location is only visible if it is set and when it is in front of the viewport
            visible: (HomeLocation.Set != 0 && Math.abs(bearing_R) < Math.PI/2)

            property real bearing_R : sceneItem.parent.circular_modulus_rad(Math.atan2(-PositionActual.East, -PositionActual.North) - yaw*Math.PI/180)
            property real elevation_R : Math.atan(-PositionActual.Down / -Math.sqrt(Math.pow(PositionActual.North,2)+Math.pow(PositionActual.East,2))) - pitch*Math.PI/180

            // Center the home location marker in the middle of the PFD
            anchors.centerIn: parent
        }
    }

    // Draw waypoint marker
    Item {
        id: waypoint
        anchors.fill: parent

        transform: [
            Translate {
                id: waypointTranslate
                x: waypoint.parent.width/2*Math.sin(nextwaypoint.bearing_R)*(Math.sin(Math.PI/2)/Math.sin(parent.fovX_D*Math.PI/180/2))
                y: -waypoint.parent.height/2*Math.sin(nextwaypoint.elevation_R)*(Math.sin(Math.PI/2)/Math.sin(parent.fovY_D*Math.PI/180/2))
            },
            Rotation {
                angle: -AttitudeActual.Roll
                origin.x : waypoint.parent.width/2
                origin.y : waypoint.parent.height/2
            }
        ]

        SvgElementImage {
            id: nextwaypoint

            elementName: "nextwaypoint"
            sceneSize: sceneItem.sceneSize

            property int activeWaypoint: WaypointActive.Index

            // When the active waypoint changes, load active the waypoint coordinates into the
            // local instance of Waypoint
            onActiveWaypointChanged: qmlWidget.exportUAVOInstance("Waypoint", activeWaypoint)

            // Waypoint is only visible when it is in front of the viewport
            visible: (Math.abs(bearing_R) < Math.PI/2 && (Waypoint.Position_North != 0 || Waypoint.Position_East != 0 || Waypoint.Position_Down != 0))

            property real bearing_R : sceneItem.parent.circular_modulus_rad(Math.atan2(Waypoint.Position_East - PositionActual.East, Waypoint.Position_North - PositionActual.North) - yaw*Math.PI/180)
            property real elevation_R : Math.atan((Waypoint.Position_Down - PositionActual.Down) / -Math.sqrt(Math.pow(Waypoint.Position_North - PositionActual.North,2)+Math.pow(Waypoint.Position_East - PositionActual.East,2))) - pitch*Math.PI/180

            // Center the home location marker in the middle of the PFD
            anchors.centerIn: parent
        }
    }

}
