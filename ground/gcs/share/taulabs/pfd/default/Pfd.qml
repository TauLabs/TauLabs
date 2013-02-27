import QtQuick 1.1
import "."
import org.TauLabs 1.0

Rectangle {
    color: "#666666"

    SvgElementImage {
        id: background
        elementName: "background"
        fillMode: Image.PreserveAspectFit
        anchors.fill: parent

        sceneSize: Qt.size(width, height)

        Item {

            // Wraps angles to -pi..pi range
            function circular_modulus_rad (x) {
                while(x > Math.PI){
                    x = x - 2*Math.PI
                }
                while(x < -Math.PI){
                    x = x + 2*Math.PI
                }

                return x
            }

            // Wraps angles to -180..180 range
            function circular_modulus_deg (x) {
                while(x > 180){
                    x = x - 360
                }
                while(x < -180){
                    x = x + 360
                }

                return x
            }

            id: sceneItem
            width: parent.paintedWidth
            height: parent.paintedHeight
            anchors.centerIn: parent
            clip: true

            // Define the field of view for the PFD. Normally this data would come
            // from the C++ code.
            property real fovX_D: 90 // In units of [deg]
            property real fovY_D: 90 // In units of [deg]

            Loader {
                id: worldLoader
                anchors.fill: parent
                source: qmlWidget.terrainEnabled ? "PfdTerrainView.qml" : "PfdWorldView.qml"
            }

            SvgElementImage {
                id: rollscale
                elementName: "rollscale"
                sceneSize: background.sceneSize

                smooth: true
                anchors.centerIn: parent
                //rotate it around the center of scene
                transform: Rotation {
                    angle: -AttitudeActual.Roll
                    origin.x : sceneItem.width/2 - x
                    origin.y : sceneItem.height/2 - y
                }
            }

            SvgElementImage {
                id: pitch_scale
                elementName: "pitch_scale"
                scale:  pitch_scale.parent.height/Math.sin(pitch_scale.parent.fovX_D*Math.PI/180/2)*Math.sin(20*Math.PI/180) / (pitch_scale.height-2*pitch_scale.border)

                smooth: true
                border: 64 //sometimes numbers are excluded from bounding rect

                anchors.centerIn: parent
                //rotate it around the center of scene
                transform: [
                    Translate {
                        id: pitchScaleTranslate
                        x: 0
                        y: -pitch_scale.parent.height/2*Math.sin((-AttitudeActual.Pitch)*Math.PI/180)*(Math.sin(Math.PI/2)/Math.sin(pitch_scale.parent.fovY_D*Math.PI/180/2))
                    },
                    Rotation {
                        angle: -AttitudeActual.Roll
                        origin.x : pitch_scale.width/2
                        origin.y : pitch_scale.height/2
                    }
                ]


            }

            SvgElementImage {
                id: pitch_desired

                elementName: "pitch-desired"
                sceneSize: sceneItem.sceneSize

                // Center the pitch desired bar in the middle of the PFD
                anchors.centerIn: parent

                transform: [
                    Translate {
                        id: pitchDesiredTranslate
                        x: 0
                        y: -pitch_desired.parent.height/2*Math.sin((StabilizationDesired.Pitch-AttitudeActual.Pitch)*Math.PI/180)*(Math.sin(Math.PI/2)/Math.sin(pitch_desired.parent.fovY_D*Math.PI/180/2))
                    },
                    Rotation {
                        angle: -AttitudeActual.Roll
                        origin.x : pitch_desired.width/2
                        origin.y : 0
                    }
                ]

                //hide if not set
                opacity: StabilizationDesired.StabilizationMode_Pitch == StabilizationDesiredType.STABILIZATIONMODE_ATTITUDE ? 1 : 
                         0
                Behavior on opacity { NumberAnimation { duration: 1000 } }
            }

            SvgElementImage {
                id: roll_desired

                elementName: "roll-desired"
                sceneSize: background.sceneSize

                smooth: true
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: rollscale.top

                //rotate it around the center of scene
                transform: Rotation {
                    angle: StabilizationDesired.Roll-AttitudeActual.Roll
                    origin.x : sceneItem.width/2 - x
                    origin.y : sceneItem.height/2 - y
                }

                //hide if not set
                opacity: StabilizationDesired.StabilizationMode_Roll == StabilizationDesiredType.STABILIZATIONMODE_ATTITUDE ? 1 : 
                         0
                Behavior on opacity { NumberAnimation { duration: 1000 } }
            }

            SvgElementImage {
                id: foreground
                elementName: "foreground"
                sceneSize: background.sceneSize

                anchors.centerIn: parent
            }

            SvgElementImage {
                id: side_slip
                elementName: "sideslip"
                sceneSize: background.sceneSize
                smooth: true

                LowPassFilter {
                    id: accelsYfiltered
                    input: Accels.y
                }
                property real sideSlip: accelsYfiltered.value

                anchors.horizontalCenter: foreground.horizontalCenter
                //0.5 coefficient is empirical to limit indicator movement
                anchors.horizontalCenterOffset: -sideSlip*width*0.5
                y: scaledBounds.y * sceneItem.height
            }

            Compass {
                anchors.fill: parent
                sceneSize: background.sceneSize
            }

            SpeedScale {
                anchors.fill: parent
                sceneSize: background.sceneSize
            }

            AltitudeScale {
                anchors.fill: parent
                sceneSize: background.sourceSize
            }

            VsiScale {
                anchors.fill: parent
                sceneSize: background.sourceSize
            }

            PfdIndicators {
                anchors.fill: parent
                sceneSize: background.sourceSize
            }
        }
    }
}
