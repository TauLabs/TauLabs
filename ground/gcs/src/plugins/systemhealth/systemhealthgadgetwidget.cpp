/**
 ******************************************************************************
 *
 * @file       systemhealthgadgetwidget.cpp
 * @author     OpenPilot Team & Edouard Lafargue Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SystemHealthPlugin System Health Plugin
 * @{
 * @brief The System Health gadget plugin
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "systemhealthgadgetwidget.h"
#include "utils/stylehelper.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "systemalarms.h"
#include <coreplugin/icore.h>
#include <QDebug>
#include <QWhatsThis>

/*
 * Initialize the widget
 */
SystemHealthGadgetWidget::SystemHealthGadgetWidget(QWidget *parent) : QGraphicsView(parent)
{
    setMinimumSize(128,64);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    setScene(new QGraphicsScene(this));

 
    m_renderer = new QSvgRenderer();
    background = new QGraphicsSvgItem();
    foreground = new QGraphicsSvgItem();
    nolink = new QGraphicsSvgItem();

    paint();

    // Now connect the widget to the SystemAlarms UAVObject
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();

    SystemAlarms* obj = SystemAlarms::GetInstance(objManager);
    connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateAlarms(UAVObject*)));

    // Listen to autopilot connection events
    TelemetryManager* telMngr = pm->getObject<TelemetryManager>();
    connect(telMngr, SIGNAL(connected()), this, SLOT(onAutopilotConnect()));
    connect(telMngr, SIGNAL(disconnected()), this, SLOT(onAutopilotDisconnect()));

    setToolTip(tr("Displays flight system errors. Click on an alarm for more information."));
}

/**
  * Hide the "No Link" overlay
  */
void SystemHealthGadgetWidget::onAutopilotConnect()
{
    nolink->setVisible(false);
}

/**
  * Show the "No Link" overlay
  */
void SystemHealthGadgetWidget::onAutopilotDisconnect()
{
    nolink->setVisible(true);
}

void SystemHealthGadgetWidget::updateAlarms(UAVObject* systemAlarm)
{
    static QList<QString> warningClean;
    // This code does not know anything about alarms beforehand, and
    // I found no efficient way to locate items inside the scene by
    // name, so it's just as simple to reset the scene:
    // And add the one with the right name.
    QGraphicsScene *m_scene = scene();
    foreach ( QGraphicsItem* item ,background->childItems()){
        m_scene->removeItem(item);
        delete item; // removeItem does _not_ delete the item.
    }

    UAVObjectField *field = systemAlarm->getField("Alarm");
    Q_ASSERT(field);
    if (field == NULL)
        return;

    for (uint i = 0; i < field->getNumElements(); ++i) {
        QString element = field->getElementNames()[i];
        QString value = field->getValue(i).toString();
        if (m_renderer->elementExists(element)) {
            QMatrix blockMatrix = m_renderer->matrixForElement(element);
            qreal startX = blockMatrix.mapRect(m_renderer->boundsOnElement(element)).x();
            qreal startY = blockMatrix.mapRect(m_renderer->boundsOnElement(element)).y();
            QString element2 = element + "-" + value;
            if (m_renderer->elementExists(element2)) {
                QGraphicsSvgItem *ind = new QGraphicsSvgItem();
                ind->setSharedRenderer(m_renderer);
                ind->setElementId(element2);
                ind->setParentItem(background);
                QTransform matrix;
                matrix.translate(startX,startY);
                ind->setTransform(matrix,false);
            } else {
                if ((value.compare("Uninitialised") != 0) && !warningClean.contains(element2))
                {
                    qDebug() << "[SystemHealth] Warning: The SystemHealth SVG does not contain a graphical element for the " << element2 << " alarm.";
                    warningClean.append(element2);
                }
            }
        } else if(!warningClean.contains(element)){
            qDebug() << "[SystemHealth] Warning: The SystemHealth SVG does not contain a graphical element for the " << element << " alarm.";
            warningClean.append(element);
        }
    }
}

SystemHealthGadgetWidget::~SystemHealthGadgetWidget()
{
   // Do nothing
}


void SystemHealthGadgetWidget::setSystemFile(QString dfn)
{
    setBackgroundBrush(QBrush(Utils::StyleHelper::baseColor()));
   if (QFile::exists(dfn)) {
       m_renderer->load(dfn);
       if(m_renderer->isValid()) {
           fgenabled = false;
           background->setSharedRenderer(m_renderer);
           background->setElementId("background");

           if (m_renderer->elementExists("foreground")) {
               foreground->setSharedRenderer(m_renderer);
               foreground->setElementId("foreground");
               foreground->setZValue(99);
               fgenabled = true;
           }
           if (m_renderer->elementExists("nolink")) {
               nolink->setSharedRenderer(m_renderer);
               nolink->setElementId("nolink");
               nolink->setZValue(100);
           }

         QGraphicsScene *l_scene = scene();
         l_scene->setSceneRect(background->boundingRect());
         fitInView(background, Qt::KeepAspectRatio );

         // Check whether the autopilot is connected already, by the way:
         ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
         UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
         TelemetryManager* telMngr = pm->getObject<TelemetryManager>();
         if (telMngr->isConnected()) {
             onAutopilotConnect();
             SystemAlarms* obj = SystemAlarms::GetInstance(objManager);
             updateAlarms(obj);
         }
       }
   }
   else
   { qDebug() <<"SystemHealthGadget: no file"; }
}

void SystemHealthGadgetWidget::paint()
{
    QGraphicsScene *l_scene = scene();
    l_scene->clear();
    l_scene->addItem(background);
    l_scene->addItem(foreground);
    l_scene->addItem(nolink);
    update();
}

void SystemHealthGadgetWidget::paintEvent(QPaintEvent *event)
{
    // Skip painting until the dial file is loaded
    if (! m_renderer->isValid()) {
        qDebug() <<"SystemHealthGadget: System file not loaded, not rendering";
        return;
    }
   QGraphicsView::paintEvent(event);
}

// This event enables the dial to be dynamically resized
// whenever the gadget is resized, taking advantage of the vector
// nature of SVG dials.
void SystemHealthGadgetWidget::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event);
    fitInView(background, Qt::KeepAspectRatio );
}

void SystemHealthGadgetWidget::mousePressEvent ( QMouseEvent * event )
{
    QGraphicsScene *graphicsScene = scene();
    if(graphicsScene){
        QPoint point = event->pos();
        bool haveAlarmItem = false;
        foreach(QGraphicsItem* sceneItem, items(point)){
            QGraphicsSvgItem *clickedItem = dynamic_cast<QGraphicsSvgItem*>(sceneItem);

            if(clickedItem){
                if((clickedItem != foreground) && (clickedItem != background)){
                    // Clicked an actual alarm. We need to set haveAlarmItem to true
                    // as two of the items in this loop will always be foreground and
                    // background. Without this flag, at some point in the loop we
                    // would always call showAllAlarmDescriptions...
                    haveAlarmItem = true;
                    QString itemId = clickedItem->elementId();
                    if(itemId.contains("OK")){
                        // No alarm set for this item
                        showAlarmDescriptionForItemId("AlarmOK", event->globalPos());
                    }else{
                        // Warning, error or critical alarm
                        showAlarmDescriptionForItemId(itemId, event->globalPos());
                    }
                }
            }
        }
        if(!haveAlarmItem){
            // Clicked foreground or background
            showAllAlarmDescriptions(event->globalPos());
        }
    }
}

void SystemHealthGadgetWidget::showAlarmDescriptionForItemId(const QString itemId, const QPoint& location){
    QFile alarmDescription(getAlarmDescriptionFileName(itemId));
    if(alarmDescription.open(QIODevice::ReadOnly | QIODevice::Text)){
        QTextStream textStream(&alarmDescription);
        QWhatsThis::showText(location, textStream.readAll());
    }
}

void SystemHealthGadgetWidget::showAllAlarmDescriptions(const QPoint& location){
    QGraphicsScene *graphicsScene = scene();
    if(graphicsScene){
        QString alarmsText;
        // Loop through all items in the scene looking for svg items that represent alarms
        foreach(QGraphicsItem* curItem, graphicsScene->items()){
            QGraphicsSvgItem* curSvgItem = dynamic_cast<QGraphicsSvgItem*>(curItem);
            if(curSvgItem && (curSvgItem != foreground) && (curSvgItem != background)){
                QString elementId = curSvgItem->elementId();
                if(!elementId.contains("OK")){
                    // Found an alarm, get its corresponding alarm html file contents
                    // and append to the cumulative string for all alarms.
                    QFile alarmDescription(getAlarmDescriptionFileName(elementId));
                    if(alarmDescription.open(QIODevice::ReadOnly | QIODevice::Text)){
                        QTextStream textStream(&alarmDescription);
                        alarmsText.append(textStream.readAll());
                        alarmDescription.close();
                    }
                }
            }
        }
        // Show alarms text if we have any
        if(alarmsText.length() > 0){
            QWhatsThis::showText(location, alarmsText);
        }
    }
}

QString SystemHealthGadgetWidget::getAlarmDescriptionFileName(const QString itemId) {
    QString alarmDescriptionFileName;
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager* objManager = pm->getObject<UAVObjectManager>();
    SystemAlarms::DataFields systemAlarmsData = SystemAlarms::GetInstance(objManager)->getData();
    if (itemId.contains("SystemConfiguration-")) {
        switch(systemAlarmsData.ConfigError) {
        case SystemAlarms::CONFIGERROR_STABILIZATION:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Error-Stabilization.html");
        break;
        case SystemAlarms::CONFIGERROR_MULTIROTOR:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Error-Multirotor.html");
        break;
        case SystemAlarms::CONFIGERROR_AUTOTUNE:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Error-AutoTune.html");
        break;
        case SystemAlarms::CONFIGERROR_ALTITUDEHOLD:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Error-AltitudeHold.html");
        break;
        case SystemAlarms::CONFIGERROR_POSITIONHOLD:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Error-PositionHold.html");
        break;
        case SystemAlarms::CONFIGERROR_PATHPLANNER:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Error-PathPlanner.html");
        break;
        case SystemAlarms::CONFIGERROR_UNDEFINED:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-Undefined.html");
        break;
        default:
            alarmDescriptionFileName = QString(":/systemhealth/html/SystemConfiguration-None.html");
        break;
        }
    } else if (itemId.contains("ManualControl-")) {
        switch(systemAlarmsData.ManualControl) {
        case SystemAlarms::MANUALCONTROL_SETTINGS:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-Critical-Settings.html");
        break;
        case SystemAlarms::MANUALCONTROL_NORX:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-Warning-NoRx.html");
        break;
        case SystemAlarms::MANUALCONTROL_ACCESSORY:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-Warning-Accessory.html");
        break;
        case SystemAlarms::MANUALCONTROL_ALTITUDEHOLD:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-Error-AltitudeHold.html");
        break;
        case SystemAlarms::MANUALCONTROL_PATHFOLLOWER:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-Critical-PathFollower.html");
        break;
        case SystemAlarms::MANUALCONTROL_UNDEFINED:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-Undefined.html");
        break;
        default:
            alarmDescriptionFileName = QString(":/systemhealth/html/ManualControl-None.html");
        break;
        }
    } else if (itemId.contains("StateEstimation-") || itemId.contains("Attitude-")) {
        switch(systemAlarmsData.StateEstimation) {
        case SystemAlarms::STATEESTIMATION_GYROQUEUENOTUPDATING:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-Gyro-Queue-Not-Updating.html");
        break;
        case SystemAlarms::STATEESTIMATION_ACCELEROMETERQUEUENOTUPDATING:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-Accelerometer-Queue-Not-Updating.html");
        break;
        case SystemAlarms::STATEESTIMATION_NOGPS:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-No-GPS.html");
        break;
        case SystemAlarms::STATEESTIMATION_NOMAGNETOMETER:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-No-Magnetometer.html");
        break;
        case SystemAlarms::STATEESTIMATION_NOBAROMETER:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-No-Barometer.html");
        break;
        case SystemAlarms::STATEESTIMATION_TOOFEWSATELLITES:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-Too-Few-Satellites.html");
        break;
        case SystemAlarms::STATEESTIMATION_PDOPTOOHIGH:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-PDOP-Too-High.html");
        break;
        case SystemAlarms::STATEESTIMATION_UNDEFINED:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-Undefined.html");
        break;
        case SystemAlarms::STATEESTIMATION_NOHOME:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-NoHome.html");
        break;
        default:
            alarmDescriptionFileName = QString(":/systemhealth/html/StateEstimation-None.html");
        break;
        }
    } else {
            alarmDescriptionFileName = QString(":/systemhealth/html/" + itemId + ".html");
    }
    return alarmDescriptionFileName; 
}

