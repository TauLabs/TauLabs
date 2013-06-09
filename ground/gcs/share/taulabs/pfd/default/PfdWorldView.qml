import Qt 4.7

Item {
    id: worldView

    Rectangle {
        id: world
        smooth: true

        property variant scaledBounds: svgRenderer.scaledElementBounds("pfd.svg", "world")
        width:  Math.round(sceneItem.width *scaledBounds.width /2)*2
        height: Math.round(sceneItem.height*scaledBounds.height/2)*2

        gradient: Gradient {
            GradientStop { position: 0.3;    color: "#6589E2" }
            GradientStop { position: 0.4999; color: "#AFC2F0" }
            GradientStop { position: 0.5;    color: "#A46933" }
            GradientStop { position: 0.8;    color: "black" }
        }

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
