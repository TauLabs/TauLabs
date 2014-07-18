/**
 ******************************************************************************
 *
 * @file       AntennaTracgadgetconfiguration.h
 * @author     Sami Korhonen & the OpenPilot team Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup AntennaTrackGadgetPlugin Antenna Track Gadget Plugin
 * @{
 * @brief A gadget that communicates with antenna tracker and enables basic configuration
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

#ifndef ANTENNATRACKGADGETCONFIGURATION_H
#define ANTENNATRACKGADGETCONFIGURATION_H

#include <coreplugin/iuavgadgetconfiguration.h>
#include <QtSerialPort/QSerialPort>

using namespace Core;

class AntennaTrackGadgetConfiguration : public IUAVGadgetConfiguration
{
    Q_OBJECT
    public:
        explicit AntennaTrackGadgetConfiguration(QString classId, QSettings* qSettings = 0, QObject *parent = 0);

        void setConnectionMode(QString mode) { m_connectionMode = mode; }
        QString connectionMode() { return m_connectionMode; }

        //set port configuration functions
        void setSpeed(QSerialPort::BaudRate speed) {m_defaultSpeed=speed;}
        void setDataBits(QSerialPort::DataBits databits) {m_defaultDataBits=databits;}
        void setFlow(QSerialPort::FlowControl flow) {m_defaultFlow=flow;}
        void setParity(QSerialPort::Parity parity) {m_defaultParity=parity;}
        void setStopBits(QSerialPort::StopBits stopbits) {m_defaultStopBits=stopbits;}
        void setPort(QString port){m_defaultPort=port;}
        void setTimeOut(long timeout){m_defaultTimeOut=timeout;}

        //get port configuration functions
        QString port(){return m_defaultPort;}
        QSerialPort::BaudRate speed() {return m_defaultSpeed;}
        QSerialPort::FlowControl flow() {return m_defaultFlow;}
        QSerialPort::DataBits dataBits() {return m_defaultDataBits;}
        QSerialPort::StopBits stopBits() {return m_defaultStopBits;}
        QSerialPort::Parity parity() {return m_defaultParity;}
        long timeOut(){return m_defaultTimeOut;}

        void saveConfig(QSettings* settings) const;
        IUAVGadgetConfiguration *clone();

    private:
        QString m_connectionMode;
        QString m_defaultPort;
        QSerialPort::BaudRate m_defaultSpeed;
        QSerialPort::DataBits m_defaultDataBits;
        QSerialPort::FlowControl m_defaultFlow;
        QSerialPort::Parity m_defaultParity;
        QSerialPort::StopBits m_defaultStopBits;
        long m_defaultTimeOut;

};

#endif // ANTENNATRACKGADGETCONFIGURATION_H
