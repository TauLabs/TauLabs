import Qt 4.7
import "."

Item {
    id: sceneItem
    property variant sceneSize

    SvgElementImage {
        id: compass
        elementName: "compass"
        sceneSize: sceneItem.sceneSize

        clip: true

        x: Math.floor(scaledBounds.x * sceneItem.width)
        y: Math.floor(scaledBounds.y * sceneItem.height)
        //anchors.horizontalCenter: parent.horizontalCenter

        //AttitudeActual.Yaw is converted to -180..180 range
        property real yaw : sceneItem.parent.circular_modulus_deg(AttitudeActual.Yaw)

        //split compass band to 8 parts to ensure it doesn't exceed the max texture size
        Row {
            id: compass_band_composed
            anchors.centerIn: parent
            //the band is 540 degrees wide
            anchors.horizontalCenterOffset: -compass.yaw/540*width

            Repeater {
                model: 5
                SvgElementImage {
                    id: compass_band
                    elementName: "compass-band"
                    sceneSize: background.sceneSize
                    hSliceCount: 5
                    hSlice: index
                }
            }
        }

        SvgElementImage {
            id: home_bearing

            elementName: "homewaypoint-bearing"
            sceneSize: sceneItem.sceneSize

            visible: HomeLocation.Set != 0

            property real bearing_D : Math.atan2(-PositionActual.East, -PositionActual.North)*180/Math.PI

            anchors.centerIn: parent
            //convert bearing-compass.yaw to -180..180 range as compass_band_composed
            //the band is 540 degrees wide
            anchors.horizontalCenterOffset: (sceneItem.parent.circular_modulus_deg(bearing_D-compass.yaw))/540*compass_band_composed.width
        }

        SvgElementImage {
            id: waypoint_bearing

            elementName: "nextwaypoint-bearing"
            sceneSize: sceneItem.sceneSize

            property int activeWaypoint: WaypointActive.Index
            onActiveWaypointChanged: qmlWidget.exportUAVOInstance("Waypoint", activeWaypoint)

            visible:  Waypoint.Position_North != 0 || Waypoint.Position_East != 0 || Waypoint.Position_Down != 0

            property real bearing_D : Math.atan2(Waypoint.Position_East - PositionActual.East, Waypoint.Position_North - PositionActual.North)*180/Math.PI

            anchors.centerIn: parent
            //convert bearing-compass.yaw to -180..180 range as compass_band_composed
            //the band is 540 degrees wide
            anchors.horizontalCenterOffset: (sceneItem.parent.circular_modulus_deg(bearing_D-compass.yaw))/540*compass_band_composed.width
        }
    }
}
