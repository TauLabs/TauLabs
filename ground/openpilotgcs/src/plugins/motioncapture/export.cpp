/**
 ******************************************************************************
 *
 * @file       esxport.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup MoCapPlugin Motion Capture Plugin
 * @{
 * @brief The Hardware In The Loop plugin
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


#include "qxtlogger.h"
#include "extensionsystem/pluginmanager.h"
#include "coreplugin/icore.h"
#include "coreplugin/threadmanager.h"
#include "export.h"
#include "mocapnoisegeneration.h"

volatile bool Export::isStarted = false;

const float Export::RAD2DEG = (180.0/M_PI);

Export::Export(const MocapSettings& params) :
    exportProcess(NULL),
	time(NULL),
	inSocket(NULL),
	outSocket(NULL),
	settings(params),
        updatePeriod(50),
        exportTimeout(8000),
	autopilotConnectionStatus(false),
    exportConnectionStatus(false),
	txTimer(NULL),
    exportTimer(NULL),
	name("")
{
	// move to thread
	moveToThread(Core::ICore::instance()->threadManager()->getRealTimeThread());
        connect(this, SIGNAL(myStart()), this, SLOT(onStart()),Qt::QueuedConnection);
	emit myStart();

    QTime currentTime=QTime::currentTime();
    gpsPosTime = currentTime;
    groundTruthTime = currentTime;
    gcsRcvrTime = currentTime;
    attRawTime = currentTime;
    baroAltTime = currentTime;
    airspeedActualTime=currentTime;

}

Export::~Export()
{
	if(inSocket)
	{
		delete inSocket;
		inSocket = NULL;
	}

	if(outSocket)
	{
		delete outSocket;
		outSocket = NULL;
	}

	if(txTimer)
	{
		delete txTimer;
		txTimer = NULL;
	}

    if(exportTimer)
	{
        delete exportTimer;
        exportTimer = NULL;
	}
	// NOTE: Does not currently work, may need to send control+c to through the terminal
    if (exportProcess != NULL)
	{
        //connect(exportProcess,SIGNAL(finished(int, QProcess::ExitStatus)),this,SLOT(onFinished(int, QProcess::ExitStatus)));

        exportProcess->disconnect();
        if(exportProcess->state() == QProcess::Running)
            exportProcess->kill();
        //if(exportProcess->waitForFinished())
            //emit deleteExportProcess();
        delete exportProcess;
        exportProcess = NULL;
	}
}

void Export::onDeleteExport(void)
{
	// [1]
    Export::setStarted(false);
	// [2]
    Export::Instances().removeOne(exportId);

	disconnect(this);
	delete this;
}

void Export::onStart()
{
    QMutexLocker locker(&lock);

    QThread* mainThread = QThread::currentThread();

    qDebug() << "Export Thread: "<< mainThread;

    // Get required UAVObjects
    ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager* objManager = pm->getObject<UAVObjectManager>();
    actDesired = ActuatorDesired::GetInstance(objManager);
    actCommand = ActuatorCommand::GetInstance(objManager);
    manCtrlCommand = ManualControlCommand::GetInstance(objManager);
    gcsReceiver= GCSReceiver::GetInstance(objManager);
    flightStatus = FlightStatus::GetInstance(objManager);
    posHome = HomeLocation::GetInstance(objManager);
    velActual = VelocityActual::GetInstance(objManager);
    posActual = PositionActual::GetInstance(objManager);
    baroAlt = BaroAltitude::GetInstance(objManager);
    airspeedActual = AirspeedActual::GetInstance(objManager);
    attActual = AttitudeActual::GetInstance(objManager);
    attSettings = AttitudeSettings::GetInstance(objManager);
    accels = Accels::GetInstance(objManager);
    gyros = Gyros::GetInstance(objManager);
    gpsPos = GPSPosition::GetInstance(objManager);
    gpsVel = GPSVelocity::GetInstance(objManager);
    telStats = GCSTelemetryStats::GetInstance(objManager);

    // Listen to autopilot connection events
    TelemetryManager* telMngr = pm->getObject<TelemetryManager>();
    connect(telMngr, SIGNAL(connected()), this, SLOT(onAutopilotConnect()));
    connect(telMngr, SIGNAL(disconnected()), this, SLOT(onAutopilotDisconnect()));
    //connect(telStats, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(telStatsUpdated(UAVObject*)));

    // If already connect setup autopilot
    GCSTelemetryStats::DataFields stats = telStats->getData();
    if ( stats.Status == GCSTelemetryStats::STATUS_CONNECTED )
        onAutopilotConnect();

    inSocket = new QUdpSocket();
    outSocket = new QUdpSocket();
    setupUdpPorts(settings.hostAddress,settings.inPort,settings.outPort);

        emit processOutput("\nLocal interface: " + settings.hostAddress + "\n" + \
                           "Remote interface: " + settings.remoteAddress + "\n" + \
                           "inputPort: " + QString::number(settings.inPort) + "\n" + \
                           "outputPort: " + QString::number(settings.outPort) + "\n");

        qxtLog->info("\nLocal interface: " + settings.hostAddress + "\n" + \
                     "Remote interface: " + settings.remoteAddress + "\n" + \
                     "inputPort: " + QString::number(settings.inPort) + "\n" + \
                     "outputPort: " + QString::number(settings.outPort) + "\n");

//        if(!inSocket->waitForConnected(5000))
//                emit processOutput(QString("Can't connect to %1 on %2 port!").arg(settings.hostAddress).arg(settings.inPort));
//        outSocket->connectToHost(settings.hostAddress,settings.outPort); // FG
//        if(!outSocket->waitForConnected(5000))
//                emit processOutput(QString("Can't connect to %1 on %2 port!").arg(settings.hostAddress).arg(settings.outPort));


	connect(inSocket, SIGNAL(readyRead()), this, SLOT(receiveUpdate()),Qt::DirectConnection);

	// Setup transmit timer
	txTimer = new QTimer();
	connect(txTimer, SIGNAL(timeout()), this, SLOT(transmitUpdate()),Qt::DirectConnection);
	txTimer->setInterval(updatePeriod);
	txTimer->start();
    // Setup export connection timer
    exportTimer = new QTimer();
    connect(exportTimer, SIGNAL(timeout()), this, SLOT(onExportConnectionTimeout()),Qt::DirectConnection);
    exportTimer->setInterval(exportTimeout);
    exportTimer->start();

	// setup time
	time = new QTime();
	time->start();
	current.T=0;
	current.i=0;

}

void Export::receiveUpdate()
{
    // Update connection timer and status
    exportTimer->setInterval(exportTimeout);
    exportTimer->stop();
    exportTimer->start();
    if ( !exportConnectionStatus )
    {
        exportConnectionStatus = true;
        emit exportConnected();
    }

    // Process data
    while(inSocket->hasPendingDatagrams()) {
        // Receive datagram
        QByteArray datagram;
        datagram.resize(inSocket->pendingDatagramSize());
        QHostAddress sender;
        quint16 senderPort;
        inSocket->readDatagram(datagram.data(), datagram.size(),
                               &sender, &senderPort);

        // Process incomming data
        processUpdate(datagram);
    }
}

void Export::setupObjects()
{

    if (settings.gcsReceiverEnabled) {
        setupInputObject(actCommand, settings.minOutputPeriod); //Input to the export
        setupOutputObject(gcsReceiver, settings.minOutputPeriod);
    } else if (settings.manualControlEnabled) {
        setupInputObject(actDesired, settings.minOutputPeriod); //Input to the export
    }

    if (settings.gpsPositionEnabled){
        setupOutputObject(gpsPos, settings.gpsPosRate);
        setupOutputObject(gpsVel, settings.gpsPosRate);
    }

    if (settings.groundTruthEnabled){
        setupOutputObject(posActual, settings.groundTruthRate);
        setupOutputObject(velActual, settings.groundTruthRate);
    }

    if (settings.attRawEnabled) {
        setupOutputObject(accels, settings.attRawRate);
        setupOutputObject(gyros, settings.attRawRate);
    }

    if (settings.attActualEnabled  && settings.attActualHW) {
        setupOutputObject(accels, settings.attRawRate);
        setupOutputObject(gyros, settings.attRawRate);
    }

    if (settings.attActualEnabled && !settings.attActualHW)
        setupOutputObject(attActual, settings.attActualRate);

    if(settings.airspeedActualEnabled)
        setupOutputObject(airspeedActual, settings.airspeedActualRate);

    if(settings.baroAltitudeEnabled)
        setupOutputObject(baroAlt, settings.baroAltRate);

}


void Export::setupInputObject(UAVObject* obj, quint32 updatePeriod)
{
    UAVObject::Metadata mdata;
    mdata = obj->getDefaultMetadata();

    UAVObject::SetGcsAccess(mdata, UAVObject::ACCESS_READONLY);
    UAVObject::SetGcsTelemetryAcked(mdata, false);
    UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_MANUAL);
    mdata.gcsTelemetryUpdatePeriod = 0;

    UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READWRITE);
    UAVObject::SetFlightTelemetryAcked(mdata, false);

    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = updatePeriod;

    obj->setMetadata(mdata);
}


void Export::setupWatchedObject(UAVObject *obj, quint32 updatePeriod)
{
    UAVObject::Metadata mdata;
    mdata = obj->getDefaultMetadata();

    UAVObject::SetGcsAccess(mdata, UAVObject::ACCESS_READONLY);
    UAVObject::SetGcsTelemetryAcked(mdata, false);
    UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_MANUAL);
    mdata.gcsTelemetryUpdatePeriod = 0;

    UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READWRITE);
    UAVObject::SetFlightTelemetryAcked(mdata, false);
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = updatePeriod;

    obj->setMetadata(mdata);
}


void Export::setupOutputObject(UAVObject* obj, quint32 updatePeriod)
{
	UAVObject::Metadata mdata;
	mdata = obj->getDefaultMetadata();

    UAVObject::SetGcsAccess(mdata, UAVObject::ACCESS_READWRITE);
    UAVObject::SetGcsTelemetryAcked(mdata, false);
    UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
    mdata.gcsTelemetryUpdatePeriod = updatePeriod;

    UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READONLY);
    UAVObject::SetFlightTelemetryUpdateMode(mdata,UAVObject::UPDATEMODE_MANUAL);

    obj->setMetadata(mdata);
}

void Export::onAutopilotConnect()
{
	autopilotConnectionStatus = true;
	setupObjects();
	emit autopilotConnected();
}

void Export::onAutopilotDisconnect()
{
	autopilotConnectionStatus = false;
	emit autopilotDisconnected();
}

void Export::onExportConnectionTimeout()
{
    if ( exportConnectionStatus )
	{
        exportConnectionStatus = false;
        emit exportDisconnected();
	}
}


void Export::telStatsUpdated(UAVObject* obj)
{
    Q_UNUSED(obj);

    GCSTelemetryStats::DataFields stats = telStats->getData();
	if ( !autopilotConnectionStatus && stats.Status == GCSTelemetryStats::STATUS_CONNECTED )
	{
		onAutopilotConnect();
	}
	else if ( autopilotConnectionStatus && stats.Status != GCSTelemetryStats::STATUS_CONNECTED )
	{
		onAutopilotDisconnect();
	}
}


void Export::resetInitialHomePosition(){
    once=false;
}


void Export::updateUAVOs(MocapOutput2Hardware out){

    QTime currentTime = QTime::currentTime();

    Noise noise;
    MoCapNoiseGeneration noiseSource;

    if(settings.addNoise){
        noise = noiseSource.generateNoise();
    }
    else{
        memset(&noise, 0, sizeof(Noise));
    }

    // Update attActual object
    AttitudeActual::DataFields attActualData;
    attActualData = attActual->getData();

    if (settings.attActualHW) {
        // do nothing
    } else if (settings.attActualMocap) {
        // take all data from export
        attActualData.Roll = out.roll + noise.attActualData.Roll;   //roll;
        attActualData.Pitch = out.pitch + noise.attActualData.Pitch;  // pitch
        attActualData.Yaw = out.yaw + noise.attActualData.Yaw; // Yaw
        float rpy[3];
        float quat[4];
        rpy[0] = attActualData.Roll;
        rpy[1] = attActualData.Pitch;
        rpy[2] = attActualData.Yaw;
        Utils::CoordinateConversions().RPY2Quaternion(rpy,quat);
        attActualData.q1 = quat[0];
        attActualData.q2 = quat[1];
        attActualData.q3 = quat[2];
        attActualData.q4 = quat[3];

        //Set UAVO
        attActual->setData(attActualData);
    }


    if (settings.gpsPositionEnabled) {
        if (gpsPosTime.msecsTo(currentTime) >= settings.gpsPosRate) {
            qDebug()<< " GPS time:" << gpsPosTime << ", currentTime: " << currentTime  << ", difference: "  << gpsPosTime.msecsTo(currentTime);
            // Update GPS Position objects
            GPSPosition::DataFields gpsPosData;
            memset(&gpsPosData, 0, sizeof(GPSPosition::DataFields));
            gpsPosData.Altitude = out.altitude + noise.gpsPosData.Altitude;
            gpsPosData.Heading = out.yaw + noise.gpsPosData.Heading;
            gpsPosData.Groundspeed = out.groundspeed + noise.gpsPosData.Groundspeed;
            gpsPosData.Latitude = out.latitude + noise.gpsPosData.Latitude;    //Already in *10^7 integer format
            gpsPosData.Longitude = out.longitude + noise.gpsPosData.Longitude; //Already in *10^7 integer format
            gpsPosData.GeoidSeparation = 0.0;
            gpsPosData.PDOP = 3.0;
            gpsPosData.VDOP = gpsPosData.PDOP*1.5;
            gpsPosData.Satellites = 10;
            gpsPosData.Status = GPSPosition::STATUS_FIX3D;

            gpsPos->setData(gpsPosData);

            // Update GPS Velocity.{North,East,Down}
            GPSVelocity::DataFields gpsVelData;
            memset(&gpsVelData, 0, sizeof(GPSVelocity::DataFields));
            gpsVelData.North = out.velNorth + noise.gpsVelData.North;
            gpsVelData.East = out.velEast + noise.gpsVelData.East;
            gpsVelData.Down = out.velDown + noise.gpsVelData.Down;

            gpsVel->setData(gpsVelData);

            gpsPosTime=gpsPosTime.addMSecs(settings.gpsPosRate);
        }
    }

    // Update PositionActual.{North,East,Down} && VelocityActual.{North,East,Down}
    if (settings.groundTruthEnabled) {
        if (groundTruthTime.msecsTo(currentTime) >= settings.groundTruthRate) {
            VelocityActual::DataFields velocityActualData;
            memset(&velocityActualData, 0, sizeof(VelocityActual::DataFields));
            velocityActualData.North = out.velNorth + noise.velocityActualData.North;
            velocityActualData.East = out.velEast + noise.velocityActualData.East;
            velocityActualData.Down = out.velDown + noise.velocityActualData.Down;
            velActual->setData(velocityActualData);

            // Update PositionActual.{Nort,East,Down}
            PositionActual::DataFields positionActualData;
            memset(&positionActualData, 0, sizeof(PositionActual::DataFields));
            positionActualData.North = out.posN + noise.positionActualData.North;
            positionActualData.East  = out.posE + noise.positionActualData.East;
            positionActualData.Down  = out.posD + noise.positionActualData.Down;
            posActual->setData(positionActualData);

            groundTruthTime=groundTruthTime.addMSecs(settings.groundTruthRate);
        }
    }

//    if (settings.sonarAltitude) {
//        static QTime sonarAltTime = currentTime;
//        if (sonarAltTime.msecsTo(currentTime) >= settings.sonarAltRate) {
//            SonarAltitude::DataFields sonarAltData;
//            sonarAltData = sonarAlt->getData();

//            float sAlt = settings.sonarMaxAlt;
//            // 0.35 rad ~= 20 degree
//            if ((agl < (sAlt * 2.0)) && (roll < 0.35) && (pitch < 0.35)) {
//                float x = agl * qTan(roll);
//                float y = agl * qTan(pitch);
//                float h = qSqrt(x*x + y*y + agl*agl);
//                sAlt = qMin(h, sAlt);
//            }

//            sonarAltData.Altitude = sAlt;
//            sonarAlt->setData(sonarAltData);
//            sonarAltTime = currentTime;
//        }
//    }

    // Update BaroAltitude object
    if (settings.baroAltitudeEnabled){
        if (baroAltTime.msecsTo(currentTime) >= settings.baroAltRate) {
        BaroAltitude::DataFields baroAltData;
        memset(&baroAltData, 0, sizeof(BaroAltitude::DataFields));
        baroAltData.Altitude = out.altitude + noise.baroAltData.Altitude;
        baroAltData.Temperature = out.temperature + noise.baroAltData.Temperature;
        baroAltData.Pressure = out.pressure + noise.baroAltData.Pressure;
        baroAlt->setData(baroAltData);

        baroAltTime=baroAltTime.addMSecs(settings.baroAltRate);
        }
    }

    // Update AirspeedActual object
    if (settings.airspeedActualEnabled){
        if (airspeedActualTime.msecsTo(currentTime) >= settings.airspeedActualRate) {
        AirspeedActual::DataFields airspeedActualData;
        memset(&airspeedActualData, 0, sizeof(AirspeedActual::DataFields));
        airspeedActual->setData(airspeedActualData);

        airspeedActualTime=airspeedActualTime.addMSecs(settings.airspeedActualRate);
        }
    }

    // Update raw attitude sensors
    if (settings.attRawEnabled) {
        if (attRawTime.msecsTo(currentTime) >= settings.attRawRate) {
            //Update gyroscope sensor data
            Gyros::DataFields gyroData;
            memset(&gyroData, 0, sizeof(Gyros::DataFields));
            gyroData.x = out.rollRate + noise.gyroData.x;
            gyroData.y = out.pitchRate + noise.gyroData.y;
            gyroData.z = out.yawRate + noise.gyroData.z;
            gyros->setData(gyroData);

            //Update accelerometer sensor data
            Accels::DataFields accelData;
            memset(&accelData, 0, sizeof(Accels::DataFields));
            accelData.x = out.accX + noise.accelData.x;
            accelData.y = out.accY + noise.accelData.y;
            accelData.z = out.accZ + noise.accelData.z;
            accels->setData(accelData);

            attRawTime=attRawTime.addMSecs(settings.attRawRate);
        }
    }
}
