import QtQuick 2.0

Item {
    id: container

    OsgEarth {
        id: earthView
        anchors.fill: parent

        sceneFile: qmlWidget.earthFile
        fieldOfView: 90

        yaw: AttitudeActual.Yaw
        pitch: AttitudeActual.Pitch
        roll: AttitudeActual.Roll

        latitude: qmlWidget.actualPositionUsed ?
                      GPSPosition.Latitude/10000000.0 : qmlWidget.latitude
        longitude: qmlWidget.actualPositionUsed ?
                       GPSPosition.Longitude/10000000.0 : qmlWidget.longitude
        altitude: qmlWidget.actualPositionUsed ?
                      GPSPosition.Altitude : qmlWidget.altitude
    }

    //Display the center line in the same way as with world view
    Item {
        id: world
        anchors.fill: parent

        transform: [
            Translate {
                id: pitchTranslate
                x: Math.round((world.parent.width - world.width)/2)
                y: (world.parent.height - world.height)/2+
                   world.parent.height/2*Math.sin(AttitudeActual.Pitch*Math.PI/180)*1.405
            },
            Rotation {
                angle: -AttitudeActual.Roll
                origin.x : world.parent.width/2
                origin.y : world.parent.height/2
            }
        ]

        SvgElementImage {
            id: horizont_line
            elementName: "world-centerline"
            //worldView is loaded with Loader, so background element is visible
            sceneSize: background.sceneSize
            anchors.centerIn: parent
            border: 1
            smooth: true
        }
    }
}
