/**
 ******************************************************************************
 *
 * @file       flightgearbridge.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *
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

#ifndef FGSIMULATOR_H
#define FGSIMULATOR_H_H

#include <QObject>
#include "simulator.h"

class FGSimulator: public Simulator
{
    Q_OBJECT

public:
	FGSimulator(const SimulatorSettings& params);
	~FGSimulator();

	bool setupProcess();
	void setupUdpPorts(const QString& host, int inPort, int outPort);

private slots:
    void transmitUpdate();
	void processReadyRead();

private:

    int udpCounterGCSsend; //keeps track of udp packets sent to FG
    int udpCounterFGrecv; //keeps track of udp packets received by FG

	void processUpdate(const QByteArray& data);

    //! Enum with indexes to use flight gear protocol.
    /*! FG UDP consists of comma separated data in floats, this are indexes to each piece of data on that datagram.
        Check opfgprotocol.xml file for details. */
    enum flightGearProtocol{
        FG_X_RATE       = 0,    /*!< x rate in radians per second       */
        FG_Y_RATE       = 1,    /*!< y rate in radians per second       */
        FG_Z_RATE       = 2,    /*!< z rate in radians per second       */
        FG_X_ACCEL      = 3,    /*!< x acceleration in feet per second^2*/
        FG_Y_ACCEL      = 4,    /*!< y acceleration in feet per second^2*/
        FG_Z_ACCEL      = 5,    /*!< z acceleration in feet per second^2*/
        FG_PITCH        = 6,    /*!< pitch angle in degrees             */
        FG_PITCH_RATE   = 7,    /*!< pitch rate in degrees per second   */
        FG_ROLL         = 8,    /*!< roll angle in degrees              */
        FG_ROLL_RATE    = 9,    /*!< roll rate in degrees per second    */
        FG_YAW          = 10,   /*!< yaw angle in degrees               */
        FG_YAW_RATE     = 11,   /*!< yaw rate in degrees per second     */
        FG_LATITUDE     = 12,   /*!< latitude in degrees                */
        FG_LONGITUDE    = 13,   /*!< longitude in degrees               */
        FG_HEADING      = 14,   /*!< heading in degrees                 */
        FG_ALTITUDE_MSL = 15,   /*!< MSL altitude in feet               */
        FG_ALTITUDE_AGL = 16,   /*!< AGL altitude in feet               */
        FG_GROUNDSPEED  = 17,   /*!< groundspeed in knots               */
        FG_AIRSPEED     = 18,   /*!< airspeed in knots                  */
        FG_TEMPERATURE  = 19,   /*!< temperature in degrees Celsius     */
        FG_PRESSURE     = 20,   /*!< pressure in inches of mercury      */
        FG_VEL_ACT_DOWN = 21,   /*!< velocity down in feet per second   */
        FG_VEL_ACT_EAST = 22,   /*!< velocity east in feet per second   */
        FG_VEL_ACT_NORTH= 23,   /*!< velocity north in feet per second  */
        FG_COUNTER_RECV = 24    /*!< udp packet counter                 */
    };
};

class FGSimulatorCreator : public SimulatorCreator
{
public:
	FGSimulatorCreator(const QString& classId, const QString& description)
	:  SimulatorCreator (classId,description)
	{}

	Simulator* createSimulator(const SimulatorSettings& params)
	{
		return new FGSimulator(params);
	}

};
#endif // FGSIMULATOR_H
