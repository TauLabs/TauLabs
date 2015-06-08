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

#include "pfdqmlgadgetwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "utils/svgimageprovider.h"
#include <QDebug>
#include <QSvgRenderer>
#include <QtCore/qfileinfo.h>
#include <QtCore/qdir.h>
#include <QMouseEvent>

#include <QQmlEngine>
#include <QQmlContext>
#include "stabilizationdesired.h"

PfdQmlGadgetWidget::PfdQmlGadgetWidget(QWindow *parent) :
    QQuickView(parent),
    m_actualPositionUsed(false),
    m_latitude(46.671478),
    m_longitude(10.158932),
    m_altitude(2000)
{
    setResizeMode(SizeRootObjectToView);

    objectsToExport << "VelocityActual" <<
                       "PositionActual" <<
                       "AltitudeHoldDesired" <<
                       "AttitudeActual" <<
                       "AirspeedActual" <<
                       "Accels" <<
                       "VelocityDesired" <<
                       "StabilizationDesired" <<
                       "PathDesired" <<
                       "HomeLocation" <<
                       "Waypoint" <<
                       "WaypointActive" <<
                       "GPSPosition" <<
                       "GCSTelemetryStats" <<
                       "FlightBatteryState";

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    m_objManager = pm->getObject<UAVObjectManager>();

    foreach (const QString &objectName, objectsToExport) {
        exportUAVOInstance(objectName, 0);
    }

    //to expose settings values
    engine()->rootContext()->setContextProperty("qmlWidget", this);
}

PfdQmlGadgetWidget::~PfdQmlGadgetWidget()
{
}


/**
 * @brief PfdQmlGadgetWidget::exportUAVOInstance Makes the UAVO available inside the QML. This works via the Q_PROPERTY()
 * values in the UAVO synthetic-headers
 * @param objectName UAVObject name
 * @param instId Instance ID
 */
void PfdQmlGadgetWidget::exportUAVOInstance(const QString &objectName, int instId)
{
    UAVObject* object = m_objManager->getObject(objectName, instId);
    if (object)
        engine()->rootContext()->setContextProperty(objectName, object);
    else
        qWarning() << "[PFDQML] Failed to load object" << objectName;
}


/**
 * @brief PfdQmlGadgetWidget::resetUAVOExport Makes the UAVO no longer available inside the QML.
 * @param objectName UAVObject name
 * @param instId Instance ID
 */
void PfdQmlGadgetWidget::resetUAVOExport(const QString &objectName, int instId)
{
    UAVObject* object = m_objManager->getObject(objectName, instId);
    if (object)
        engine()->rootContext()->setContextProperty(objectName, (QObject*)NULL);
    else
        qWarning() << "Failed to load object" << objectName;
}

void PfdQmlGadgetWidget::setQmlFile(QString fn)
{
    m_qmlFileName = fn;

    engine()->removeImageProvider("svg");
    SvgImageProvider *svgProvider = new SvgImageProvider(fn);
    engine()->addImageProvider("svg", svgProvider);

    engine()->clearComponentCache();

    //it's necessary to allow qml side to query svg element position
    engine()->rootContext()->setContextProperty("svgRenderer", svgProvider);
    engine()->setBaseUrl(QUrl::fromLocalFile(fn));

    qDebug() << Q_FUNC_INFO << fn;
    setSource(QUrl::fromLocalFile(fn));

    foreach(const QQmlError &error, errors()) {
        qDebug() << error.description();
    }
}

//Switch between PositionActual UAVObject position
//and pre-defined latitude/longitude/altitude properties
void PfdQmlGadgetWidget::setActualPositionUsed(bool arg)
{
    if (m_actualPositionUsed != arg) {
        m_actualPositionUsed = arg;
        emit actualPositionUsedChanged(arg);
    }
}

void PfdQmlGadgetWidget::setSettingsMap(const QVariantMap &settings)
{
    engine()->rootContext()->setContextProperty("settings", settings);
}

void PfdQmlGadgetWidget::mouseReleaseEvent(QMouseEvent *event)
{
    // Reload the schene on the middle mouse button click.
    if (event->button() == Qt::MiddleButton) {
        setQmlFile(m_qmlFileName);
    }

    QQuickView::mouseReleaseEvent(event);
}

void PfdQmlGadgetWidget::setLatitude(double arg)
{
    //not sure qFuzzyCompare is accurate enough for geo coordinates
    if (m_latitude != arg) {
        m_latitude = arg;
        emit latitudeChanged(arg);
    }
}

void PfdQmlGadgetWidget::setLongitude(double arg)
{
    if (m_longitude != arg) {
        m_longitude = arg;
        emit longitudeChanged(arg);
    }
}

void PfdQmlGadgetWidget::setAltitude(double arg)
{
    if (!qFuzzyCompare(m_altitude,arg)) {
        m_altitude = arg;
        emit altitudeChanged(arg);
    }
}

