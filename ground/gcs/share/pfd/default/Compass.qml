import QtQuick 2.0
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

            visible: HomeLocation.Set !== 0

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

            visible:  Waypoint.Position_North !== 0 || Waypoint.Position_East !== 0 || Waypoint.Position_Down !== 0

            property real bearing_D : Math.atan2(Waypoint.Position_East - PositionActual.East, Waypoint.Position_North - PositionActual.North)*180/Math.PI

            anchors.centerIn: parent
            //convert bearing-compass.yaw to -180..180 range as compass_band_composed
            //the band is 540 degrees wide
            anchors.horizontalCenterOffset: (sceneItem.parent.circular_modulus_deg(bearing_D-compass.yaw))/540*compass_band_composed.width
        }

        SvgElementImage {
            id: unfiltered_compass_bearing

            elementName: "unfiltered-compass-bearing"
            sceneSize: sceneItem.sceneSize

            visible: Magnetometer.x !== 0 || Magnetometer.y !==0 || Magnetometer.z !==0

            // Calculate unfiltered magnetic heading, corrected for magnetic inclination
            function calculateHeading()
            {
                var cP=Math.cos(AttitudeActual.Yaw*Math.PI/180*0)
                var sP=Math.sin(AttitudeActual.Yaw*Math.PI/180*0)

                var cT=Math.cos(AttitudeActual.Pitch*Math.PI/180)
                var sT=Math.sin(AttitudeActual.Pitch*Math.PI/180)

                var cF=Math.cos(AttitudeActual.Roll*Math.PI/180)
                var sF=Math.sin(AttitudeActual.Roll*Math.PI/180)

                var Rbe00 = cT*cP
                var Rbe01 = cT*sP
                var Rbe02 = -sT
                var Rbe10 = sF*sT*cP - cF*sP
                var Rbe11 = sF*sT*sP + cF*cP
                var Rbe12 = cT*sF
                var Rbe20 = cF*sT*cP + sF*sP
                var Rbe21 = cF*sT*sP - sF*cP
                var Rbe22 = cT*cF

                // Rotate body frame magnetometer measurment into Earth frame.
                var mag_N = Rbe00*Magnetometer.x + Rbe10*Magnetometer.y + Rbe20*Magnetometer.z
                var mag_E = Rbe01*Magnetometer.x + Rbe11*Magnetometer.y + Rbe21*Magnetometer.z

                // Calculate compass heading, relative to magnetic North
                var magnetic_heading_R = Math.atan2(-mag_E, mag_N)

                // Calculate local magnetic declination
                var magnetic_inclination_R = Math.atan2(-HomeLocation.Be_1, HomeLocation.Be_0)

                // Return magnetic compass heading, relative to true North
                return magnetic_heading_R - magnetic_inclination_R
            }


            property real heading_D : calculateHeading()*180/Math.PI

            anchors.centerIn: parent
            //convert bearing-compass.yaw to -180..180 range as compass_band_composed
            //the band is 540 degrees wide
            anchors.horizontalCenterOffset: (sceneItem.parent.circular_modulus_deg(heading_D-compass.yaw))/540*compass_band_composed.width

        }

    }
}
