/**
 ******************************************************************************
 * @file       kmlexport.cpp
 * @brief Exports log data to KML
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup KmlExportPlugin
 * @{
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


#ifndef KMLEXPORT_H
#define KMLEXPORT_H

#include <QIODevice>
#include <QTime>
#include <QTimer>
#include <QDebug>
#include <QBuffer>
#include <math.h>

#include "kml/base/file.h"
#include "kml/dom.h"
#include "kml/engine.h"

#include "./uavtalk/uavtalk.h"

#include "airspeedactual.h"
#include "attitudeactual.h"
#include "homelocation.h"
#include "gpsposition.h"
#include "positionactual.h"
#include "velocityactual.h"

using namespace kmldom;


// This struct holds the 4D LLA-Velocity coordinates
struct LLAVCoordinates
{
    double latitude;
    double longitude;
    double altitude;
    double groundspeed; //in [m/s]
};

/**
 * @class KmlExport generates a KML file showing the flight path from a UAVTalk
 * log path that is viewable in Google Earth.
 */
class KmlExport : public QObject
{
    Q_OBJECT
public:
    explicit KmlExport(QString inputFileName, QString outputFileName);
    qint64 bytesAvailable() const;
    qint64 bytesToWrite() { return logFile.bytesToWrite(); }
    bool open();
    void setFileName(QString name) { logFile.setFileName(name); }

    bool preparseLogFile();
    bool stopExport();
    bool exportToKML();

private slots:
    void gpsPositionUpdated(UAVObject *);
    void homeLocationUpdated(UAVObject *);
    void positionActualUpdated(UAVObject *);

signals:
    void readReady();
    void replayStarted();
    void replayFinished();

protected:
    QFile logFile;

private:
    QList<quint32> timestampBuffer;
    QList<quint32> timestampPos;

    UAVTalk *kmlTalk;

    AirspeedActual *airspeedActual;
    AttitudeActual *attitudeActual;
    GPSPosition *gpsPosition;
    HomeLocation *homeLocation;
    PositionActual *positionActual;
    VelocityActual *velocityActual;

    GPSPosition::DataFields gpsPositionData;
    HomeLocation::DataFields homeLocationData;

    DocumentPtr document;
    FolderPtr trackFolder;
    FolderPtr timestampFolder;
    KmlFactory *factory;

    QString outputFileName;
    LLAVCoordinates oldPoint;
    quint32 timeStamp;
    quint32 lastPlacemarkTime;
    QString informationString;
    QVector<CoordinatesPtr> wallAxes;
    static QString dateTimeFormat;

    void parseLogFile();
    StylePtr createGroundTrackStyle();
    StyleMapPtr createWallAxesStyle();
    StyleMapPtr createCustomBalloonStyle();
    PlacemarkPtr CreateLineStringPlacemark(const LLAVCoordinates &startPoint, const LLAVCoordinates &endPoint, quint32 newPlacemarkTime);
    PlacemarkPtr createTimespanPlacemark(const LLAVCoordinates &point, quint32 lastPlacemarkTime, quint32 newPlacemarkTime);

    kmlbase::Color32 mapVelocity2Color(double velocity, quint8 alpha = 255);
};

//! Jet color map, as defined by matlab. Generated with `jet(256)`.
#define COLORMAP_JET { \
    {0, 0, 0.5156}, \
    {0, 0, 0.5312}, \
    {0, 0, 0.5469}, \
    {0, 0, 0.5625}, \
    {0, 0, 0.5781}, \
    {0, 0, 0.5938}, \
    {0, 0, 0.6094}, \
    {0, 0, 0.6250}, \
    {0, 0, 0.6406}, \
    {0, 0, 0.6562}, \
    {0, 0, 0.6719}, \
    {0, 0, 0.6875}, \
    {0, 0, 0.7031}, \
    {0, 0, 0.7188}, \
    {0, 0, 0.7344}, \
    {0, 0, 0.7500}, \
    {0, 0, 0.7656}, \
    {0, 0, 0.7812}, \
    {0, 0, 0.7969}, \
    {0, 0, 0.8125}, \
    {0, 0, 0.8281}, \
    {0, 0, 0.8438}, \
    {0, 0, 0.8594}, \
    {0, 0, 0.8750}, \
    {0, 0, 0.8906}, \
    {0, 0, 0.9062}, \
    {0, 0, 0.9219}, \
    {0, 0, 0.9375}, \
    {0, 0, 0.9531}, \
    {0, 0, 0.9688}, \
    {0, 0, 0.9844}, \
    {0, 0, 1.0000}, \
    {0, 0.0156, 1.0000}, \
    {0, 0.0312, 1.0000}, \
    {0, 0.0469, 1.0000}, \
    {0, 0.0625, 1.0000}, \
    {0, 0.0781, 1.0000}, \
    {0, 0.0938, 1.0000}, \
    {0, 0.1094, 1.0000}, \
    {0, 0.1250, 1.0000}, \
    {0, 0.1406, 1.0000}, \
    {0, 0.1562, 1.0000}, \
    {0, 0.1719, 1.0000}, \
    {0, 0.1875, 1.0000}, \
    {0, 0.2031, 1.0000}, \
    {0, 0.2188, 1.0000}, \
    {0, 0.2344, 1.0000}, \
    {0, 0.2500, 1.0000}, \
    {0, 0.2656, 1.0000}, \
    {0, 0.2812, 1.0000}, \
    {0, 0.2969, 1.0000}, \
    {0, 0.3125, 1.0000}, \
    {0, 0.3281, 1.0000}, \
    {0, 0.3438, 1.0000}, \
    {0, 0.3594, 1.0000}, \
    {0, 0.3750, 1.0000}, \
    {0, 0.3906, 1.0000}, \
    {0, 0.4062, 1.0000}, \
    {0, 0.4219, 1.0000}, \
    {0, 0.4375, 1.0000}, \
    {0, 0.4531, 1.0000}, \
    {0, 0.4688, 1.0000}, \
    {0, 0.4844, 1.0000}, \
    {0, 0.5000, 1.0000}, \
    {0, 0.5156, 1.0000}, \
    {0, 0.5312, 1.0000}, \
    {0, 0.5469, 1.0000}, \
    {0, 0.5625, 1.0000}, \
    {0, 0.5781, 1.0000}, \
    {0, 0.5938, 1.0000}, \
    {0, 0.6094, 1.0000}, \
    {0, 0.6250, 1.0000}, \
    {0, 0.6406, 1.0000}, \
    {0, 0.6562, 1.0000}, \
    {0, 0.6719, 1.0000}, \
    {0, 0.6875, 1.0000}, \
    {0, 0.7031, 1.0000}, \
    {0, 0.7188, 1.0000}, \
    {0, 0.7344, 1.0000}, \
    {0, 0.7500, 1.0000}, \
    {0, 0.7656, 1.0000}, \
    {0, 0.7812, 1.0000}, \
    {0, 0.7969, 1.0000}, \
    {0, 0.8125, 1.0000}, \
    {0, 0.8281, 1.0000}, \
    {0, 0.8438, 1.0000}, \
    {0, 0.8594, 1.0000}, \
    {0, 0.8750, 1.0000}, \
    {0, 0.8906, 1.0000}, \
    {0, 0.9062, 1.0000}, \
    {0, 0.9219, 1.0000}, \
    {0, 0.9375, 1.0000}, \
    {0, 0.9531, 1.0000}, \
    {0, 0.9688, 1.0000}, \
    {0, 0.9844, 1.0000}, \
    {0, 1.0000, 1.0000}, \
    {0.0156, 1.0000, 0.9844}, \
    {0.0312, 1.0000, 0.9688}, \
    {0.0469, 1.0000, 0.9531}, \
    {0.0625, 1.0000, 0.9375}, \
    {0.0781, 1.0000, 0.9219}, \
    {0.0938, 1.0000, 0.9062}, \
    {0.1094, 1.0000, 0.8906}, \
    {0.1250, 1.0000, 0.8750}, \
    {0.1406, 1.0000, 0.8594}, \
    {0.1562, 1.0000, 0.8438}, \
    {0.1719, 1.0000, 0.8281}, \
    {0.1875, 1.0000, 0.8125}, \
    {0.2031, 1.0000, 0.7969}, \
    {0.2188, 1.0000, 0.7812}, \
    {0.2344, 1.0000, 0.7656}, \
    {0.2500, 1.0000, 0.7500}, \
    {0.2656, 1.0000, 0.7344}, \
    {0.2812, 1.0000, 0.7188}, \
    {0.2969, 1.0000, 0.7031}, \
    {0.3125, 1.0000, 0.6875}, \
    {0.3281, 1.0000, 0.6719}, \
    {0.3438, 1.0000, 0.6562}, \
    {0.3594, 1.0000, 0.6406}, \
    {0.3750, 1.0000, 0.6250}, \
    {0.3906, 1.0000, 0.6094}, \
    {0.4062, 1.0000, 0.5938}, \
    {0.4219, 1.0000, 0.5781}, \
    {0.4375, 1.0000, 0.5625}, \
    {0.4531, 1.0000, 0.5469}, \
    {0.4688, 1.0000, 0.5312}, \
    {0.4844, 1.0000, 0.5156}, \
    {0.5000, 1.0000, 0.5000}, \
    {0.5156, 1.0000, 0.4844}, \
    {0.5312, 1.0000, 0.4688}, \
    {0.5469, 1.0000, 0.4531}, \
    {0.5625, 1.0000, 0.4375}, \
    {0.5781, 1.0000, 0.4219}, \
    {0.5938, 1.0000, 0.4062}, \
    {0.6094, 1.0000, 0.3906}, \
    {0.6250, 1.0000, 0.3750}, \
    {0.6406, 1.0000, 0.3594}, \
    {0.6562, 1.0000, 0.3438}, \
    {0.6719, 1.0000, 0.3281}, \
    {0.6875, 1.0000, 0.3125}, \
    {0.7031, 1.0000, 0.2969}, \
    {0.7188, 1.0000, 0.2812}, \
    {0.7344, 1.0000, 0.2656}, \
    {0.7500, 1.0000, 0.2500}, \
    {0.7656, 1.0000, 0.2344}, \
    {0.7812, 1.0000, 0.2188}, \
    {0.7969, 1.0000, 0.2031}, \
    {0.8125, 1.0000, 0.1875}, \
    {0.8281, 1.0000, 0.1719}, \
    {0.8438, 1.0000, 0.1562}, \
    {0.8594, 1.0000, 0.1406}, \
    {0.8750, 1.0000, 0.1250}, \
    {0.8906, 1.0000, 0.1094}, \
    {0.9062, 1.0000, 0.0938}, \
    {0.9219, 1.0000, 0.0781}, \
    {0.9375, 1.0000, 0.0625}, \
    {0.9531, 1.0000, 0.0469}, \
    {0.9688, 1.0000, 0.0312}, \
    {0.9844, 1.0000, 0.0156}, \
    {1.0000, 1.0000, 0}, \
    {1.0000, 0.9844, 0}, \
    {1.0000, 0.9688, 0}, \
    {1.0000, 0.9531, 0}, \
    {1.0000, 0.9375, 0}, \
    {1.0000, 0.9219, 0}, \
    {1.0000, 0.9062, 0}, \
    {1.0000, 0.8906, 0}, \
    {1.0000, 0.8750, 0}, \
    {1.0000, 0.8594, 0}, \
    {1.0000, 0.8438, 0}, \
    {1.0000, 0.8281, 0}, \
    {1.0000, 0.8125, 0}, \
    {1.0000, 0.7969, 0}, \
    {1.0000, 0.7812, 0}, \
    {1.0000, 0.7656, 0}, \
    {1.0000, 0.7500, 0}, \
    {1.0000, 0.7344, 0}, \
    {1.0000, 0.7188, 0}, \
    {1.0000, 0.7031, 0}, \
    {1.0000, 0.6875, 0}, \
    {1.0000, 0.6719, 0}, \
    {1.0000, 0.6562, 0}, \
    {1.0000, 0.6406, 0}, \
    {1.0000, 0.6250, 0}, \
    {1.0000, 0.6094, 0}, \
    {1.0000, 0.5938, 0}, \
    {1.0000, 0.5781, 0}, \
    {1.0000, 0.5625, 0}, \
    {1.0000, 0.5469, 0}, \
    {1.0000, 0.5312, 0}, \
    {1.0000, 0.5156, 0}, \
    {1.0000, 0.5000, 0}, \
    {1.0000, 0.4844, 0}, \
    {1.0000, 0.4688, 0}, \
    {1.0000, 0.4531, 0}, \
    {1.0000, 0.4375, 0}, \
    {1.0000, 0.4219, 0}, \
    {1.0000, 0.4062, 0}, \
    {1.0000, 0.3906, 0}, \
    {1.0000, 0.3750, 0}, \
    {1.0000, 0.3594, 0}, \
    {1.0000, 0.3438, 0}, \
    {1.0000, 0.3281, 0}, \
    {1.0000, 0.3125, 0}, \
    {1.0000, 0.2969, 0}, \
    {1.0000, 0.2812, 0}, \
    {1.0000, 0.2656, 0}, \
    {1.0000, 0.2500, 0}, \
    {1.0000, 0.2344, 0}, \
    {1.0000, 0.2188, 0}, \
    {1.0000, 0.2031, 0}, \
    {1.0000, 0.1875, 0}, \
    {1.0000, 0.1719, 0}, \
    {1.0000, 0.1562, 0}, \
    {1.0000, 0.1406, 0}, \
    {1.0000, 0.1250, 0}, \
    {1.0000, 0.1094, 0}, \
    {1.0000, 0.0938, 0}, \
    {1.0000, 0.0781, 0}, \
    {1.0000, 0.0625, 0}, \
    {1.0000, 0.0469, 0}, \
    {1.0000, 0.0312, 0}, \
    {1.0000, 0.0156, 0}, \
    {1.0000, 0, 0}, \
    {0.9844, 0, 0}, \
    {0.9688, 0, 0}, \
    {0.9531, 0, 0}, \
    {0.9375, 0, 0}, \
    {0.9219, 0, 0}, \
    {0.9062, 0, 0}, \
    {0.8906, 0, 0}, \
    {0.8750, 0, 0}, \
    {0.8594, 0, 0}, \
    {0.8438, 0, 0}, \
    {0.8281, 0, 0}, \
    {0.8125, 0, 0}, \
    {0.7969, 0, 0}, \
    {0.7812, 0, 0}, \
    {0.7656, 0, 0}, \
    {0.7500, 0, 0}, \
    {0.7344, 0, 0}, \
    {0.7188, 0, 0}, \
    {0.7031, 0, 0}, \
    {0.6875, 0, 0}, \
    {0.6719, 0, 0}, \
    {0.6562, 0, 0}, \
    {0.6406, 0, 0}, \
    {0.6250, 0, 0}, \
    {0.6094, 0, 0}, \
    {0.5938, 0, 0}, \
    {0.5781, 0, 0}, \
    {0.5625, 0, 0}, \
    {0.5469, 0, 0}, \
    {0.5312, 0, 0}, \
    {0.5156, 0, 0}, \
    {0.5000, 0, 0} }

#endif // KMLEXPORT_H
