/**
 ******************************************************************************
 *
 * @file       exporter.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup MoCapPlugin Motion Capture Plugin
 * @{
 * @brief Motion capture plugin which communicates via UDP
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

#ifndef EXPORTER_H
#define EXPORTER_H

#include <QObject>
#include <QUdpSocket>
#include <QTimer>
#include <QProcess>
#include <qmath.h>

#include "qscopedpointer.h"
#include "uavtalk/telemetrymanager.h"
#include "uavobjectmanager.h"
#include "ui_mocapwidget.h"

#include "accels.h"
#include "actuatorcommand.h"
#include "actuatordesired.h"
#include "attitudeactual.h"
#include "attitudesettings.h"
#include "airspeedactual.h"
#include "baroaltitude.h"
#include "flightstatus.h"
#include "gcsreceiver.h"
#include "gcstelemetrystats.h"
#include "gpsposition.h"
#include "gpsvelocity.h"
#include "gyros.h"
#include "homelocation.h"
#include "manualcontrolcommand.h"
#include "positionactual.h"
#include "sonaraltitude.h"
#include "velocityactual.h"

#include "utils/coordinateconversions.h"
#include "physical_constants.h"

/**
 * just imagine this was a class without methods and all public properties
 */
typedef struct _FLIGHT_PARAM {

    // time
    float T;
    float dT;
    unsigned int i;

    // speed (relative)
    float ias;
    float cas;
    float tas;
    float groundspeed;

    // position (absolute)
    float X;
    float Y;
    float Z;

    // speed (absolute)
    float dX;
    float dY;
    float dZ;

    // acceleration (absolute)
    float ddX;
    float ddY;
    float ddZ;

    //angle
    float azimuth;
    float pitch;
    float roll;

    //rotation speed
    float dAzimuth;
    float dPitch;
    float dRoll;

} FLIGHT_PARAM;

typedef struct _CONNECTION
{
    QString exporterId;
    QString binPath;
    QString dataPath;
    QString hostAddress;
    QString remoteAddress;
    int outPort;
    int inPort;
    bool addNoise;
    QString latitude;
    QString longitude;

//    bool homeLocation;

    bool attRawEnabled;
    quint8 attRawRate;

    bool attActualEnabled;
    bool attActualHW;
    bool attActualMocap;
    quint16 attActualRate;

    bool baroAltitudeEnabled;
    quint16 baroAltRate;

    bool groundTruthEnabled;
    quint16 groundTruthRate;

    bool gpsPositionEnabled;
    quint16 gpsPosRate;

    bool inputCommand;
    bool gcsReceiverEnabled;
    bool manualControlEnabled;
    quint16 minOutputPeriod;

    bool airspeedActualEnabled;
    quint16 airspeedActualRate;

} MocapSettings;


struct MocapOutput2Hardware{
    float longitude;  //[deg]
    float latitude;   //[deg]
    float altitude;   //[m]
    float posN;       //[m]
    float posE;       //[m]
    float posD;       //[m]
    float velNorth;   //[m/s]
    float velEast;    //[m/s]
    float velDown;    //[m/s]
    float accX;       //[m/s^2]
    float accY;       //[m/s^2]
    float accZ;       //[m/s^2]
    float roll;
    float pitch;
    float yaw;
    float rollRate;     //[deg/s]
    float pitchRate;     //[deg/s]
    float yawRate;     //[deg/s]
    float groundspeed; //[m/s]
    float pressure;
    float temperature;
    float delT;
};

class Exporter : public QObject
{
    Q_OBJECT

public:
    Exporter(const MocapSettings& params);
    virtual ~Exporter();

    bool isAutopilotConnected() const { return autopilotConnectionStatus; }
    bool isExporterConnected() const { return exporterConnectionStatus; }
    QString Name() const { return name; }
    void setName(QString str) { name = str; }

    QString ExporterId() const { return exporterId; }
    void setExporterId(QString str) { exporterId = str; }



    static bool IsStarted() { return isStarted; }
    static void setStarted(bool val) { isStarted = val; }
    static QStringList& Instances() { return Exporter::instances; }
    static void setInstance(const QString& str) { Exporter::instances.append(str); }

    virtual void stopProcess() {}
    virtual void setupUdpPorts(const QString& host, int inPort, int outPort) { Q_UNUSED(host) Q_UNUSED(inPort) Q_UNUSED(outPort)}

    void resetInitialHomePosition();
    void updateUAVOs(MocapOutput2Hardware out);

signals:
    void autopilotConnected();
    void autopilotDisconnected();
    void exporterConnected();
    void exporterDisconnected();
    void processOutput(QString str);
    void deleteExporterProcess();
    void myStart();
public slots:
    Q_INVOKABLE virtual bool setupProcess() { return true;}
private slots:
    void onStart();
    //void transmitUpdate();
    void receiveUpdate();
    void onAutopilotConnect();
    void onAutopilotDisconnect();
    void onExporterConnectionTimeout();
    void telStatsUpdated(UAVObject* obj);
    Q_INVOKABLE void onDeleteExporter(void);

    virtual void transmitUpdate() = 0;
    virtual void processUpdate(const QByteArray& data) = 0;

protected:
    QProcess* exporterProcess;
    QTime* time;
    QUdpSocket* inSocket;//(new QUdpSocket());
    QUdpSocket* outSocket;

    ActuatorCommand* actCommand;
    ActuatorDesired* actDesired;
    ManualControlCommand* manCtrlCommand;
    FlightStatus* flightStatus;
    BaroAltitude* baroAlt;
    AirspeedActual* airspeedActual;
    AttitudeActual* attActual;
    AttitudeSettings* attSettings;
    VelocityActual* velActual;
    GPSPosition* gpsPos;
    GPSVelocity* gpsVel;
    PositionActual* posActual;
    HomeLocation* posHome;
    Accels* accels;
    Gyros*  gyros;
    GCSTelemetryStats* telStats;
    GCSReceiver* gcsReceiver;

    MocapSettings settings;

    FLIGHT_PARAM current;
    FLIGHT_PARAM old;
    QMutex lock;

private:
    bool once;

    int updatePeriod;
    int exporterTimeout;
    volatile bool autopilotConnectionStatus;
    volatile bool exporterConnectionStatus;
    QTimer* txTimer;
    QTimer* exporterTimer;

    QTime attRawTime;
    QTime gpsPosTime;
    QTime groundTruthTime;
    QTime baroAltTime;
    QTime gcsRcvrTime;
    QTime airspeedActualTime;

    QString name;
    QString exporterId;
    volatile static bool isStarted;
    static QStringList instances;
    //QList<QScopedPointer<UAVDataObject> > requiredUAVObjects;
    void setupOutputObject(UAVObject* obj, quint32 updatePeriod);
    void setupInputObject(UAVObject* obj, quint32 updatePeriod);
    void setupWatchedObject(UAVObject *obj, quint32 updatePeriod);
    void setupObjects();

    Ui_MoCapWidget *widget;
};



class MocapCreator
{
public:
    MocapCreator(QString id, QString descr) :
        classId(id),
        description(descr)
    {}
    virtual ~MocapCreator() {}

    QString ClassId() const {return classId;}
    QString Description() const {return description;}

    virtual Exporter* createExporter(const MocapSettings& params, Ui_MoCapWidget *widget) = 0;

private:
    QString classId;
    QString description;
};

#endif // EXPORTER_H
