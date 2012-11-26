/**
 ******************************************************************************
 *
 * @file       mocapconfiguration.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup HITLPlugin HITL Plugin
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

#include "mocapconfiguration.h"

MoCapConfiguration::MoCapConfiguration(QString classId, QSettings* qSettings, QObject *parent) :
    IUAVGadgetConfiguration(classId, parent)
{

    //Default settings values
    settings.exporterId = "";
    settings.binPath = "";
    settings.dataPath = "";
    settings.manualControlEnabled = true;
    settings.addNoise = false;
    settings.hostAddress = "127.0.0.1";
    settings.remoteAddress = "127.0.0.1";
    settings.outPort = 0;
    settings.inPort = 0;
    settings.latitude = "";
    settings.longitude = "";

    settings.attRawEnabled       = false;
    settings.attRawRate          = 20;

    settings.attActualEnabled    = true;
    settings.attActualHW         = false;
    settings.attActualMocap      = true;
    settings.attActualRate       = 50;

    settings.gpsPositionEnabled  = false;
    settings.gpsPosRate          = 100;

    settings.groundTruthEnabled  = false;
    settings.groundTruthRate     = 100;

    settings.inputCommand        = false;
    settings.gcsReceiverEnabled  = false;
    settings.manualControlEnabled= false;
    settings.minOutputPeriod     = 100;

    settings.airspeedActualEnabled= false;
    settings.airspeedActualRate  = 100;


    // if a saved configuration exists load it, and overwrite defaults
    if (qSettings != 0) {

        settings.exporterId         = qSettings->value("exporterId").toString();
        settings.binPath             = qSettings->value("binPath").toString();
        settings.dataPath            = qSettings->value("dataPath").toString();

        settings.hostAddress         = qSettings->value("hostAddress").toString();
        settings.remoteAddress       = qSettings->value("remoteAddress").toString();
        settings.outPort             = qSettings->value("outPort").toInt();
        settings.inPort              = qSettings->value("inPort").toInt();

        settings.latitude            = qSettings->value("latitude").toString();
        settings.longitude           = qSettings->value("longitude").toString();
        settings.addNoise            = qSettings->value("noiseCheckBox").toBool();

        settings.inputCommand        = qSettings->value("inputCommand").toBool();
        if(settings.inputCommand){
            settings.gcsReceiverEnabled  = qSettings->value("gcsReceiverEnabled").toBool();
            settings.manualControlEnabled= qSettings->value("manualControlEnabled").toBool();
        }
        else{
            settings.gcsReceiverEnabled  = false;
            settings.manualControlEnabled= false;
        }

        settings.attRawEnabled       = qSettings->value("attRawEnabled").toBool();
        settings.attRawRate          = qSettings->value("attRawRate").toInt();

        settings.attActualEnabled    = qSettings->value("attActualEnabled").toBool();
        if(settings.attActualEnabled)
        {
            settings.attActualHW     = qSettings->value("attActualHW").toBool();
            settings.attActualMocap  = qSettings->value("attActualMocap").toBool();
        }
        else
        {
            settings.attActualHW        = false;
            settings.attActualMocap     = false;
        }
        settings.attActualRate      = qSettings->value("attActualRate").toInt();

        settings.baroAltitudeEnabled = qSettings->value("baroAltitudeEnabled").toBool();
        settings.baroAltRate         = qSettings->value("baroAltRate").toInt();

        settings.gpsPositionEnabled  = qSettings->value("gpsPositionEnabled").toBool();
        settings.gpsPosRate          = qSettings->value("gpsPosRate").toInt();

        settings.groundTruthEnabled  = qSettings->value("groundTruthEnabled").toBool();
        settings.groundTruthRate     = qSettings->value("groundTruthRate").toInt();

        settings.minOutputPeriod     = qSettings->value("minOutputPeriod").toInt();

        settings.airspeedActualEnabled=qSettings->value("airspeedActualEnabled").toBool();
        settings.airspeedActualRate  = qSettings->value("airspeedActualRate").toInt();
    }
}

IUAVGadgetConfiguration *MoCapConfiguration::clone()
{
    MoCapConfiguration *m = new MoCapConfiguration(this->classId());

	m->settings = settings;
    return m;
}

 /**
  * Saves a configuration.
  *
  */
void MoCapConfiguration::saveConfig(QSettings* qSettings) const {
    qSettings->setValue("exporterId", settings.exporterId);
    qSettings->setValue("binPath", settings.binPath);
    qSettings->setValue("dataPath", settings.dataPath);

    qSettings->setValue("hostAddress", settings.hostAddress);
    qSettings->setValue("remoteAddress", settings.remoteAddress);
    qSettings->setValue("outPort", settings.outPort);
    qSettings->setValue("inPort", settings.inPort);

    qSettings->setValue("latitude", settings.latitude);
    qSettings->setValue("longitude", settings.longitude);
    qSettings->setValue("addNoise", settings.addNoise);

    qSettings->setValue("inputCommand", settings.inputCommand);
    qSettings->setValue("gcsReceiverEnabled", settings.gcsReceiverEnabled);
    qSettings->setValue("manualControlEnabled", settings.manualControlEnabled);

    qSettings->setValue("attRawEnabled", settings.attRawEnabled);
    qSettings->setValue("attRawRate", settings.attRawRate);
    qSettings->setValue("attActualEnabled", settings.attActualEnabled);
    qSettings->setValue("attActualHW", settings.attActualHW);
    qSettings->setValue("attActualMocap", settings.attActualMocap);
    qSettings->setValue("attActualRate", settings.attActualRate);
    qSettings->setValue("baroAltitudeEnabled", settings.baroAltitudeEnabled);
    qSettings->setValue("baroAltRate", settings.baroAltRate);
    qSettings->setValue("gpsPositionEnabled", settings.gpsPositionEnabled);
    qSettings->setValue("gpsPosRate", settings.gpsPosRate);
    qSettings->setValue("groundTruthEnabled", settings.groundTruthEnabled);
    qSettings->setValue("groundTruthRate", settings.groundTruthRate);
    qSettings->setValue("minOutputPeriod", settings.minOutputPeriod);

    qSettings->setValue("airspeedActualEnabled", settings.airspeedActualEnabled);
    qSettings->setValue("airspeedActualRate", settings.airspeedActualRate);
}

