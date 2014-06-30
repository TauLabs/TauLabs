// import QtQuick 1.0 // to target S60 5th Edition or Maemo 5
import QtQuick 1.1

Item {
    id: container
    width: 100
    height: 62

    signal clicked(string url)

    Text {
        id: header
        text: "GitHub Activity"
        width: parent.width
        color: "#44515c"
        font {
            pointSize: 14
            weight: Font.Bold
        }
    }

    ListView {
        id: view
        width: parent.width
        anchors { top: header.bottom; topMargin: 14; bottom: parent.bottom }
        model: xmlModel
        delegate: listDelegate
        clip: true
    }

    ScrollDecorator {
        flickableItem: view
    }

    XmlListModel {
        id: xmlModel
        source: "http://github.com/TauLabs/TauLabs/commits/next.atom"
        query: "/feed/entry"
        namespaceDeclarations: "declare namespace media=\"http://search.yahoo.com/mrss/\"; declare default element namespace \"http://www.w3.org/2005/Atom\" ;"
        XmlRole { name: "title"; query: "title/string()" }
        XmlRole { name: "author"; query: "author/name/string()" }
        XmlRole { name: "authoruri"; query: "author/uri/string()" }
        XmlRole { name: "link"; query: "link/@href/string()" }
        XmlRole { name: "thumbnail"; query: 'media:thumbnail/@url/string()' }
    }

    Component {
        id: listDelegate
        Item  {
            width: view.width
            height: column.height + 8

            Row {
                id: column
                spacing: 4
                Image {
                    id: recipeImage
                    width: 30; height: 30
                    source: thumbnail
                    MouseArea {
                        anchors.fill: parent  //...to cover the whole image
                        onClicked: Qt.openUrlExternally(authoruri)
                    }
                }
                Column
                {
                    Text {
                        text: author
                        font.italic: true
                        width: view.width
                        textFormat: text.indexOf("&") > 0 ? Text.StyledText : Text.PlainText
                        elide: Text.ElideRight
                        color: mouseArea.containsMouse ? "#224d81" : "black"
                        MouseArea {
                            anchors.fill: parent  //...to cover the whole image
                            onClicked: Qt.openUrlExternally(authoruri)
                        }
                    }
                    Text {
                        text: title
                        width: view.width
                        elide: Text.ElideRight
                        font.bold: true
                        color: mouseArea.containsMouse ? "#224d81" : "black"
                        MouseArea {
                            anchors.fill: parent  //...to cover the whole image
                            onClicked: Qt.openUrlExternally(link)
                        }
                    }



                }

            }
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
