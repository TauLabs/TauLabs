// import QtQuick 1.0 // to target S60 5th Edition or Maemo 5
import QtQuick 1.1

Item {
    property alias sourceSize: background.sourceSize
    width: sourceSize.width
    height: 300

    BorderImage {
        id: background
        anchors.fill: parent

        border { left: 30; top: 30; right: 30; bottom: 30 }
        source: "images/welcome-news-bg.png"
    }

    GitHubNewsPanel {
        id: gitHubNewsPanel
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: parent.width*0.35
        anchors.margins: 32

        onClicked: welcomePlugin.openUrl(url)
    }

    //better to use image instead
    Rectangle {
        id: separator
        width: 1
        height: parent.height*0.7
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: gitHubNewsPanel.right
        anchors.margins: 16
        color: "#A0A0A0"
    }
    ForumNewsPanel {
        id: forumNewsPanel
        anchors.left: gitHubNewsPanel.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: parent.width*0.35
        anchors.margins: 32

        onClicked: welcomePlugin.openUrl(url)
    }

    //better to use image instead
    Rectangle {
        id: separator2
        width: 1
        height: parent.height*0.7
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: forumNewsPanel.right
        anchors.margins: 16
        color: "#A0A0A0"
    }
    SitesPanel {
        transformOrigin: Item.Center
        anchors.left: forumNewsPanel.right
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: 32

        onClicked: welcomePlugin.openUrl(url)
    }
}
