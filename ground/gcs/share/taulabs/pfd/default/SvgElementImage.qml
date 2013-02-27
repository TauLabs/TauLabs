import QtQuick 1.1

Image {
    id: sceneItem
    property variant sceneSize
    property string elementName
    property string svgFileName: "pfd.svg"
    property int vSlice: 0
    property int vSliceCount: 0
    property int hSlice: 0
    property int hSliceCount: 0
    // The border property is useful to extend the area of the image a
    // bit, so it looks anti-aliased when rotated
    property int border: 0
    property variant scaledBounds: svgRenderer.scaledElementBounds(svgFileName, elementName)

    sourceSize.width: Math.round(sceneSize.width*scaledBounds.width)
    sourceSize.height: Math.round(sceneSize.height*scaledBounds.height)

    // Generates a source string and loads the corresponding image
    function generateSource() {
        var params = ""
        if (hSliceCount > 1)
            params += "hslice="+hSlice+":"+hSliceCount+";"
        if (vSliceCount > 1)
            params += "vslice="+vSlice+":"+vSliceCount+";"
        if (border > 0)
            params += "border="+border+";"

        if (params != "")
            params = "?" + params

        // Load source image
        source = "image://svg/"+svgFileName+"!"+elementName+params
    }

    Component.onCompleted: { generateSource() }
}
