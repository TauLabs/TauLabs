/*
    This file is part of AutoQuad.

    AutoQuad is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AutoQuad is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AutoQuad.  If not, see <http://www.gnu.org/licenses/>.

    Copyright © 2011, 2012, 2013  Bill Nesbitt
*/

#include "aq.h"
#include "ublox.h"
#include "gps.h"
#include "aq_timer.h"
#include "imu.h"
#include "config.h"
#include "util.h"
#include "rtc.h"
#include "filer.h"
#include "supervisor.h"
#include "comm.h"
#include <CoOS.h>
#include <string.h>


static void ubloxTxChecksumReset(void) {
    ubloxData.ubloxTxCK_A = 0;
    ubloxData.ubloxTxCK_B = 0;
}

static void ubloxRxChecksumReset(void) {
    ubloxData.ubloxRxCK_A = 0;
    ubloxData.ubloxRxCK_B = 0;
}

static void ubloxRxChecksum(unsigned char c) {
    ubloxData.ubloxRxCK_A += c;
    ubloxData.ubloxRxCK_B += ubloxData.ubloxRxCK_A;
}

static void ubloxTxChecksum(uint8_t c) {
    ubloxData.ubloxTxCK_A += c;
    ubloxData.ubloxTxCK_B += ubloxData.ubloxTxCK_A;
}

static void ubloxWriteU1(unsigned char c) {
    serialWrite(gpsData.gpsPort, c);
    ubloxTxChecksum(c);
}

static void ubloxWriteI1(signed char c) {
    serialWrite(gpsData.gpsPort, (unsigned char)c);
    ubloxTxChecksum(c);
}

static void ubloxWriteU2(unsigned short int x) {
    ubloxWriteU1(x);
    ubloxWriteU1(x>>8);
}

static void ubloxWriteI2(signed short int x) {
    ubloxWriteU1(x);
    ubloxWriteU1(x>>8);
}

static void ubloxWriteU4(unsigned long int x) {
    ubloxWriteU1(x);
    ubloxWriteU1(x>>8);
    ubloxWriteU1(x>>16);
    ubloxWriteU1(x>>24);
}

static void ubloxWriteI4(signed long int x) {
    ubloxWriteU1(x);
    ubloxWriteU1(x>>8);
    ubloxWriteU1(x>>16);
    ubloxWriteU1(x>>24);
}

static void ubloxSendPreamble(void) {
    ubloxWriteU1(0xb5);	// u
    ubloxWriteU1(0x62);	// b

    ubloxTxChecksumReset();
}

static void ubloxEnableMessage(unsigned char c, unsigned char i, unsigned char rate) {
    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_CFG_CLASS);  // CFG
    ubloxWriteU1(UBLOX_CFG_MSG);    // MSG

    ubloxWriteU1(0x03);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    ubloxWriteU1(c);		    // class
    ubloxWriteU1(i);		    // id
    ubloxWriteU1(rate);		    // rate

    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_A);
    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_B);
    yield(UBLOX_WAIT_MS);
}

static void ubloxSetRate(unsigned short int ms) {
    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_CFG_CLASS);  // CFG
    ubloxWriteU1(UBLOX_CFG_RTATE);  // RTATE

    ubloxWriteU1(0x06);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    ubloxWriteU2(ms);		    // rate
    ubloxWriteU2(0x01);		    // cycles
    ubloxWriteU2(0x01);		    // timeRef	0 == UTC, 1 == GPS time

    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_A);
    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_B);
    yield(UBLOX_WAIT_MS);
}

static void ubloxSetMode(void) {
    int i;

    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_CFG_CLASS);  // CFG
    ubloxWriteU1(UBLOX_CFG_NAV5);   // NAV5

    ubloxWriteU1(0x24);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    ubloxWriteU1(0b0000101);	    // mask LSB (fixMode, dyn)
    ubloxWriteU1(0x00);		    // mask MSB (reserved)
    ubloxWriteU1(0x06);		    // dynModel (6 == airborne < 1g, 8 == airborne < 4g)
    ubloxWriteU1(0x02);		    // fixMode (2 == 3D only)

    // the rest of the packet is ignored due to the above mask
    for (i = 0; i < 32; i++)
	ubloxWriteU1(0x00);

    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_A);
    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_B);
    yield(UBLOX_WAIT_MS);
}

static void ubloxSetTimepulse(void) {
    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_CFG_CLASS);  // CFG
    ubloxWriteU1(UBLOX_CFG_TP);	    // TP

    ubloxWriteU1(0x14);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    ubloxWriteU4(1000000);	    // interval (us)
    ubloxWriteU4(100000);	    // length (us)
#ifdef GPS_LATENCY
    ubloxWriteI1(0x00);		    // config setting (0 == off)
#else
    ubloxWriteI1(0x01);		    // config setting (1 == +polarity)
#endif
    ubloxWriteU1(0x01);		    // alignment reference time (GPS)
    ubloxWriteU1(0x00);		    // bitmask (syncmode 0)
    ubloxWriteU1(0x00);		    // reserved
    ubloxWriteI2(0x00);		    // antenna delay
    ubloxWriteI2(0x00);		    // rf group delay
    ubloxWriteI4(0x00);		    // user delay

    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_A);
    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_B);
    yield(UBLOX_WAIT_MS);
}

static void ubloxSetSBAS(uint8_t enable) {
    // second bit of mode field is diffCorr
    enable = (enable > 0);

    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_CFG_CLASS);  // CFG
    ubloxWriteU1(UBLOX_CFG_SBAS);   // SBAS

    ubloxWriteU1(0x08);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    ubloxWriteU1(enable);	    // enable
    ubloxWriteU1(0b011);	    // mode
    ubloxWriteU1(3);		    // # SBAS tracking channels
    ubloxWriteU1(0);
    ubloxWriteU4(UBLOX_SBAS_AUTO);  // ANY SBAS system

    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_A);
    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_B);
    yield(UBLOX_WAIT_MS);
}

static void ubloxPollVersion(void) {
    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_MON_CLASS);  // MON
    ubloxWriteU1(UBLOX_MON_VER);    // VER

    ubloxWriteU1(0x00);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_A);
    serialWrite(gpsData.gpsPort, ubloxData.ubloxTxCK_B);
    yield(UBLOX_WAIT_MS);
}

static void ubloxVersionSpecific(int ver) {
    if (ver > 6) {
	// 10Hz for ver 7+
	ubloxSetRate((uint16_t)100);
	// SBAS screwed up on v7 modules w/ v1 firmware
	ubloxSetSBAS(0);					// disable SBAS
    }
    else {
	// 5Hz
	ubloxSetRate((uint16_t)200);
    }
}

void ubx_cfg_send_configuration(void) {
    ubloxSetTimepulse();
    ubloxEnableMessage(UBLOX_NAV_CLASS, UBLOX_NAV_VALNED, 1);	// NAV VALNED
    ubloxEnableMessage(UBLOX_NAV_CLASS, UBLOX_NAV_POSLLH, 1);	// NAV POSLLH
    ubloxEnableMessage(UBLOX_TIM_CLASS, UBLOX_TIM_TP, 1);	// TIM TP
    ubloxEnableMessage(UBLOX_NAV_CLASS, UBLOX_NAV_DOP, 5);	// NAV DOP
    ubloxEnableMessage(UBLOX_AID_CLASS, UBLOX_AID_REQ, 1);	// AID REQ
    ubloxEnableMessage(UBLOX_NAV_CLASS, UBLOX_NAV_TIMEUTC, 5);	// NAV TIMEUTC
#ifdef GPS_DO_RTK
    ubloxEnableMessage(UBLOX_RXM_CLASS, UBLOX_RXM_RAW, 1);	// RXM RAW
    ubloxEnableMessage(UBLOX_RXM_CLASS, UBLOX_RXM_SFRB, 1);	// RXM SFRB
#endif
#ifdef GPS_DEBUG
    ubloxEnableMessage(UBLOX_NAV_CLASS, UBLOX_NAV_SVINFO, 1);	// NAV SVINFO
    ubloxEnableMessage(UBLOX_NAV_CLASS, UBLOX_NAV_SBAS, 1);	// NAV SBAS
    ubloxEnableMessage(UBLOX_MON_CLASS, UBLOX_MON_HW, 1);	// MON HW
#endif

    ubloxSetMode();						// 3D, airborne
    ubloxPollVersion(); 
    ubloxVersionSpecific(ubloxData.hwVer);
}

//! Set the output baudrate to 230400
void ubx_cfg_set_baudrate(uint16_t baud_rate) {
    // UBX,41 msg
    // 1 - portID
    // 0007 - input protocol (all)
    // 0001 - output protocol (ubx only)
    // 230400 - baudrate
    // 0 - no attempt to autobaud
    // 0x18 - baudrate
    const uint8_t * msg = "$PUBX,41,1,0007,0001,230400,0*18\n";

    // TODO: send message
}

