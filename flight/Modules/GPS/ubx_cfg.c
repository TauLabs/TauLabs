/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GSPModule GPS Module
 * @brief Process GPS information
 * @{
 *
 * @file       ubx_cfg.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Include file for UBX configuration
 * @see        The GNU Public License (GPL) Version 3
 *
 * Copyright © 2011, 2012, 2013  Bill Nesbitt
 *
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

#include "openpilot.h"
#include "pios_com.h"

/*
 * The format of UBX Packets is documented in the UBX Protocol
 * documentation and is summarized below
 *
 * 1 byte - SYNC_CHAR1 (UBLOX_SYNC1 = 0xB5)
 * 1 byte - SYNC_CHAR2 (UBLOX_SYNC2 = 0x62)
 * 1 byte - CLASS 
 * 1 byte - ID
 * 2 byte - length (little endian)
 * N bytes - payload
 * 2 bytes - checksum
 *
 * The checksum is calculated from the class to the last byte of
 * the payload
 */
#define UBLOX_SYNC1     0xB5
#define UBLOX_SYNC2     0x62

#define UBLOX_NAV_CLASS     0x01
#define UBLOX_RXM_CLASS     0x02
#define UBLOX_CFG_CLASS     0x06
#define UBLOX_MON_CLASS     0x0a
#define UBLOX_AID_CLASS     0x0b
#define UBLOX_TIM_CLASS     0x0d


#define UBLOX_NAV_POSLLH    0x02
#define UBLOX_NAV_DOP       0x04
#define UBLOX_NAV_VALNED    0x12
#define UBLOX_NAV_TIMEUTC   0x21
#define UBLOX_NAV_SBAS      0x32
#define UBLOX_NAV_SVINFO    0x30

#define UBLOX_AID_REQ       0x00

#define UBLOX_RXM_RAW       0x10
#define UBLOX_RXM_SFRB      0x11

#define UBLOX_MON_VER       0x04
#define UBLOX_MON_HW        0x09

#define UBLOX_TIM_TP        0x01

#define UBLOX_CFG_MSG       0x01
#define UBLOX_CFG_TP        0x07
#define UBLOX_CFG_RTATE     0x08
#define UBLOX_CFG_SBAS      0x16
#define UBLOX_CFG_NAV5      0x24

#define UBLOX_SBAS_AUTO     0x00000000
#define UBLOX_SBAS_WAAS     0x0004E004
#define UBLOX_SBAS_EGNOS    0x00000851
#define UBLOX_SBAS_MSAS     0x00020200
#define UBLOX_SBAS_GAGAN    0x00000108

#define UBLOX_MAX_PAYLOAD   384
#define UBLOX_WAIT_MS       20

uintptr_t gps_tx_comm;

uint8_t ubloxTxCK_A, ubloxTxCK_B;

static void ubloxTxChecksumReset(void) {
    ubloxTxCK_A = 0;
    ubloxTxCK_B = 0;
}

static void ubloxTxChecksum(uint8_t c) {
    ubloxTxCK_A += c;
    ubloxTxCK_B += ubloxTxCK_A;
}

static void ubloxWriteU1(unsigned char c) {
    PIOS_COM_SendChar(gps_tx_comm, c);
    ubloxTxChecksum(c);
}

static void ubloxWriteI1(signed char c) {
    PIOS_COM_SendChar(gps_tx_comm, (unsigned char)c);
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
    PIOS_COM_SendChar(gps_tx_comm, UBLOX_SYNC1);	// u
    PIOS_COM_SendChar(gps_tx_comm, UBLOX_SYNC2);	// b

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

    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);

    vTaskDelay(TICKS2MS(UBLOX_WAIT_MS));
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

    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);

    vTaskDelay(TICKS2MS(UBLOX_WAIT_MS));
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

    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);

    vTaskDelay(TICKS2MS(UBLOX_WAIT_MS));
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

    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);

    vTaskDelay(TICKS2MS(UBLOX_WAIT_MS));
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

    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);

    vTaskDelay(TICKS2MS(UBLOX_WAIT_MS));
}

static void ubloxPollVersion(void) {
    ubloxSendPreamble();

    ubloxWriteU1(UBLOX_MON_CLASS);  // MON
    ubloxWriteU1(UBLOX_MON_VER);    // VER

    ubloxWriteU1(0x00);		    // length lsb
    ubloxWriteU1(0x00);		    // length msb

    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);

    vTaskDelay(TICKS2MS(UBLOX_WAIT_MS));
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

//! Send a stream of data followed by checksum
static void ubx_cfg_send_checksummed(uintptr_t gps_port,
    const uint8_t *dat, uint16_t len)
{
    // Calculate checksum
    ubloxTxChecksumReset();
    for (uint16_t i = 0; i < len; i++) {
        ubloxTxChecksum(dat[i]);
    }

    // Send buffer followed by checksum
    PIOS_COM_SendBuffer(gps_port, dat, len);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_tx_comm, ubloxTxCK_B);  
}

void ubx_cfg_send_configuration(uintptr_t gps_port)
{

    if (!gps_port)
        return;

    gps_tx_comm = gps_port;
    vTaskDelay(MS2TICKS(UBLOX_WAIT_MS));

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

    // Hardcoded version. The poll version method should fetch the
    // data but we need to link to that.
    ubloxVersionSpecific(6);
}

//! Set the output baudrate to 230400
void ubx_cfg_set_baudrate(uint16_t baud_rate)
{
    // UBX,41 msg
    // 1 - portID
    // 0007 - input protocol (all)
    // 0001 - output protocol (ubx only)
    // 230400 - baudrate
    // 0 - no attempt to autobaud
    // 0x18 - baudrate
    const char * msg = "$PUBX,41,1,0007,0001,230400,0*18\n";

    // Attempt to configure at common baud rates
    PIOS_COM_ChangeBaud(gps_tx_comm, 4800);
    PIOS_COM_SendString(gps_tx_comm, msg);
    PIOS_COM_ChangeBaud(gps_tx_comm, 9600);
    PIOS_COM_SendString(gps_tx_comm, msg);
    PIOS_COM_ChangeBaud(gps_tx_comm, 57600);
    PIOS_COM_SendString(gps_tx_comm, msg);
    PIOS_COM_ChangeBaud(gps_tx_comm, 240400);
}

/**
 * @}
 * @}
 */