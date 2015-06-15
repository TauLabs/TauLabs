/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GSPModule GPS Module
 * @brief Process GPS information
 * @{
 *
 * @file       ubx_cfg.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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
#include "pios_thread.h"

#if !defined(PIOS_GPS_MINIMAL)

#include "GPS.h"
#include "UBX.h"

#include "gpsposition.h"
#include "modulesettings.h"

/*
 * The format of UBX Packets is documented in the UBX Protocol
 * documentation and is summarized below. Links to the original
 * documentation can be found here:
 *
 * http://www.u-blox.com/en/download/documents-a-resources/u-blox-6-gps-modules-resources.html
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
#define UBLOX_NAV_STATUS    0x03
#define UBLOX_NAV_DOP       0x04
#define UBLOX_NAV_SOL       0x06
#define UBLOX_NAV_VELNED    0x12
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
#define UBLOX_CFG_CFG       0x09
#define UBLOX_CFG_SBAS      0x16
#define UBLOX_CFG_NAV5      0x24
#define UBLOX_CFG_GNSS      0x3E

#define UBLOX_SBAS_AUTO     0x00000000
// TODO: Reverify these constants-- seems they have more bits set than they
// should.
#define UBLOX_SBAS_WAAS     0x0004E004
#define UBLOX_SBAS_EGNOS    0x00000851
#define UBLOX_SBAS_MSAS     0x00020200
#define UBLOX_SBAS_GAGAN    0x00000108

#define UBLOX_DYN_PORTABLE   0
#define UBLOX_DYN_STATIONARY 2
#define UBLOX_DYN_PED        3
#define UBLOX_DYN_AUTOMOTIVE 4
#define UBLOX_DYN_SEA        5
#define UBLOX_DYN_AIR1G      6
#define UBLOX_DYN_AIR2G      7
#define UBLOX_DYN_AIR4G      8

#define UBLOX_GNSSID_GPS     0
#define UBLOX_GNSSID_SBAS    1
#define UBLOX_GNSSID_BEIDOU  3
#define UBLOX_GNSSID_QZSS    5
#define UBLOX_GNSSID_GLONASS 6

#define UBLOX_MAX_PAYLOAD   384
#define UBLOX_WAIT_MS       20

uint8_t ubloxTxCK_A, ubloxTxCK_B;
static char *gps_rx_buffer;

static void ubx_cfg_send_checksummed(uintptr_t gps_port, const uint8_t *dat, uint16_t len);

//! Reset the TX checksum calculation
static void ubloxTxChecksumReset(void) {
    ubloxTxCK_A = 0;
    ubloxTxCK_B = 0;
}

//! Update the checksum calculation
static void ubloxTxChecksum(uint8_t c) {
    ubloxTxCK_A += c;
    ubloxTxCK_B += ubloxTxCK_A;
}

//! Enable the selected UBX message at the specified rate
static void ubx_cfg_enable_message(uintptr_t gps_port,
    uint8_t c, uint8_t i, uint8_t rate) {

    const uint8_t msg[] = {
        UBLOX_CFG_CLASS,       // CFG
        UBLOX_CFG_MSG,         // MSG
        0x03,                  // length lsb
        0x00,                  // length msb
        c,                     // class
        i,                     // id
        rate,                  // rate
    };
    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}

//! Set the rate of all messages
static void ubx_cfg_set_rate(uintptr_t gps_port, uint16_t ms) {

    const uint8_t msg[] = {
        UBLOX_CFG_CLASS,       // CFG
        UBLOX_CFG_RTATE,       // RTATE
        0x06,                  // length lsb
        0x00,                  // length msb
        ms,                    // rate lsb
        ms >> 8,               // rate msb
        0x01,                  // cycles
        0x00,
        0x01,                  // timeref 1 = GPS time
        0x00,
    };
    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}

//! Configure the navigation mode and minimum fix
static void ubx_cfg_set_mode(uintptr_t gps_port,
        ModuleSettingsGPSDynamicsModeOptions dyn_mode) {
    uint8_t dyn_const;

    // Omitted: Stationary and at sea.  At sea assumes sea level, we basically
    // never want that.

    switch (dyn_mode) {
        case MODULESETTINGS_GPSDYNAMICSMODE_PORTABLE:
            dyn_const = UBLOX_DYN_PORTABLE;
            break;
        case MODULESETTINGS_GPSDYNAMICSMODE_PEDESTRIAN:
            dyn_const = UBLOX_DYN_PED;
            break;
        case MODULESETTINGS_GPSDYNAMICSMODE_AUTOMOTIVE:
            dyn_const = UBLOX_DYN_AUTOMOTIVE;
            break;
        case MODULESETTINGS_GPSDYNAMICSMODE_AIRBORNE1G:
            dyn_const = UBLOX_DYN_AIR1G;
            break;
        case MODULESETTINGS_GPSDYNAMICSMODE_AIRBORNE2G:
        default:
            dyn_const = UBLOX_DYN_AIR2G;
            break;
        case MODULESETTINGS_GPSDYNAMICSMODE_AIRBORNE4G:
            dyn_const = UBLOX_DYN_AIR4G;
            break;
    }

    const uint8_t msg[] = {
        UBLOX_CFG_CLASS,       // CFG
        UBLOX_CFG_NAV5,        // NAV5 mode
        0x24,                  // length lsb - 36 bytes
        0x00,                  // length msb
        0b0000101,             // mask LSB (fixMode, dyn)
        0x00,                  // mask MSB (reserved)
        dyn_const,             // dynamic model (7 - airborne < 2g)
        0x02,                  // fixmode (2 - 3D only)
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,       // padded with 32 zeros
    };
    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}

//! Configure the timepulse output pin
static void ubx_cfg_set_timepulse(uintptr_t gps_port) {
    const uint8_t TP_POLARITY = 1;
    const uint32_t int_us = 1000000;
    const uint32_t len_us = 100000;

    const uint8_t msg[] = {
        UBLOX_CFG_CLASS,       // CFG
        UBLOX_CFG_TP,          // TP
        0x14,                  // length lsb
        0x00,                  // length msb
        int_us & 0xff,
        int_us >> 8 & 0xff,   // interval (us)
        int_us >> 16 & 0xff,
        int_us >> 24 & 0xff,
        len_us & 0xff,
        len_us >> 8 & 0xff,     // length (us)
        len_us >> 16 & 0xff,
        len_us >> 24 & 0xff,
        TP_POLARITY,           // polarity - zero off
        0x01,                  // 1 - GPS time
        0x00,                  // bitmask
        0x00,                  // reserved
        0x00, 0x00,            // antenna delay
        0x00, 0x00,            // rf group delay
        0, 0, 0, 0             // user delay
    };
    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}

//! Enable or disable SBAS satellites
static void ubx_cfg_set_sbas(uintptr_t gps_port,
        ModuleSettingsGPSSBASConstellationOptions sbas_const) {
    bool enable = sbas_const != MODULESETTINGS_GPSSBASCONSTELLATION_NONE;

    uint32_t sv_mask;

    switch (sbas_const) {
        case MODULESETTINGS_GPSSBASCONSTELLATION_WAAS:
            sv_mask = UBLOX_SBAS_WAAS;
            break;
        case MODULESETTINGS_GPSSBASCONSTELLATION_EGNOS:
            sv_mask = UBLOX_SBAS_EGNOS;
            break;
        case MODULESETTINGS_GPSSBASCONSTELLATION_MSAS:
            sv_mask = UBLOX_SBAS_MSAS;
            break;
        case MODULESETTINGS_GPSSBASCONSTELLATION_GAGAN:
            sv_mask = UBLOX_SBAS_GAGAN;
            break;
        case MODULESETTINGS_GPSSBASCONSTELLATION_ALL:
        case MODULESETTINGS_GPSSBASCONSTELLATION_NONE:
        default:
            sv_mask = UBLOX_SBAS_AUTO;
            break;
    }

    const uint8_t msg[] = {
        UBLOX_CFG_CLASS, // CFG
        UBLOX_CFG_SBAS,  // SBAS
        0x08,            // length lsb
        0x00,            // length msb
        enable,          // enable flag
        0b011,           // mode
        3,               // # SBAS tracking channels
        0,
        sv_mask,
        sv_mask >> 8,
        sv_mask >> 16,
        sv_mask >> 24,
    };

    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}


/**
 * Erases the internal storage for message and navigation
 * configuration. Does not do anything until UBX reboot 
 * though.
 */
static void ubx_cfg_clear_cfg(uintptr_t gps_port) {

    // Reset the messges and navigation settings
    const uint32_t MASK = 0b00001110;
    const uint8_t msg[] = {
        UBLOX_CFG_CLASS, // CFG
        UBLOX_CFG_CFG,   // CFG-CFG
        0x0C,            // length lsb
        0x00,            // length msb
        MASK,            // clear mask (U4)
        MASK >> 8,
        MASK >> 16,
        MASK >> 24,
        0,0,0,0,         // load mask
        0,0,0,0          // save mask
    };

    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}

//! Request a MON-VER message with the firmware version
static void ubx_cfg_poll_version(uintptr_t gps_port) {
    const uint8_t msg[] = {UBLOX_MON_CLASS, UBLOX_MON_VER, 0x00, 0x00};
    ubx_cfg_send_checksummed(gps_port, msg, sizeof(msg));
}

static void ubx_cfg_set_constellation(uintptr_t gps_port, 
        ModuleSettingsGPSConstellationOptions constellation,
        ModuleSettingsGPSSBASConstellationOptions sbas_const) {
    // Needs to handle 20 length for one constellation + SBAS, length 28 for
    // second constellation (second constellation)
    uint8_t len = 20;
    uint8_t config_blks = 2;
    uint8_t gnss_id = UBLOX_GNSSID_GPS;

    uint8_t sbas_chan = 3;

    bool sbas_enabled = (sbas_const != MODULESETTINGS_GPSSBASCONSTELLATION_NONE);

    // Don't save channels for SBAS when we're not using it.
    if (!sbas_enabled) sbas_chan = 0;

    switch (constellation) {
        case MODULESETTINGS_GPSCONSTELLATION_ALL:
            len = 28;   // Defaults-- just send the glonass data too.
            break;
        case MODULESETTINGS_GPSCONSTELLATION_GLONASS:
            gnss_id = UBLOX_GNSSID_GLONASS;
                        // Just send the first two blocks, only GLONASS const
            break;
        case MODULESETTINGS_GPSCONSTELLATION_GPS:
        default:
            // Nothing to see here.  Defaults are good.
            break;
    }

    // XXX TODO send configuration
    const uint8_t msg[] = {
        UBLOX_CFG_CLASS, // CFG
        UBLOX_CFG_GNSS,  // GNSS
        len,             // length lsb
        0x00,            // length msb
        0,               // msgver = 0   offset=0
        0,               // numTrkChHw (ro)
        0,               // numTrkChUse (ro)
        config_blks,     // numConfigBlks -- 1 or 2 constellations?

        gnss_id,         // ID of first constellation
        16,              // Minimum number of channels to reserve.  m8
                         // has 72 channels, so saving 16 for GPS is no big
                         // deal.
        72,              // maximum number of channels used
        0,               // reserved1
        1,               // flags, 1 here means enable
        0,
        1,               // flags, sigcfgmask, 1 sane for all sat systems
        0,

        UBLOX_GNSSID_SBAS, // This chunk is always for SBAS/DGPS
        sbas_chan,       // How many to save for SBAS?
        sbas_chan,       // Maximum sbas channels
        0,               // reserved1
        sbas_enabled,    // flags, 1 here means enable
        0,
        1,               // flags, sigcfgmask, SBAS L1CA
        0,

        // If this next one is used, it's always GLONASS.
        UBLOX_GNSSID_GLONASS,
        4,               // Minimum num channels reserved -- if enabled, always
                         // save at least a few for glonass acquisition
        72,              // Maximum sbas channels
        0,               // reserved1
        1,               // flags, 1 here means enable
        0,
        1,               // flags, sigcfgmask, GLONASS L1OF
        0
    };

    // The 4 offset here is for the header.
    ubx_cfg_send_checksummed(gps_port, msg, len + 4);
}

//! Apply firmware version specific configuration tweaks
static void ubx_cfg_version_specific(uintptr_t gps_port, uint8_t ver,
        ModuleSettingsGPSConstellationOptions constellation,
        ModuleSettingsGPSSBASConstellationOptions sbas_const) {
    // Enable satellite-based differential GPS.
    ubx_cfg_set_sbas(gps_port, sbas_const);

    if (ver >= 8) {
        // 10Hz for ver 8, unless 'ALL' constellations in which case we
        // are 5Hz
        // TODO: Detect modules where we can ask for 18/10Hz instead
        if (constellation == MODULESETTINGS_GPSCONSTELLATION_ALL) {
            ubx_cfg_set_rate(gps_port, (uint16_t)200);
        } else {
            ubx_cfg_set_rate(gps_port, (uint16_t)100);
        }

        ubx_cfg_set_constellation(gps_port, constellation, sbas_const);
    } else if (ver == 7) {
        // 10Hz for ver 7
        ubx_cfg_set_rate(gps_port, (uint16_t)100);
    } else if (ver == 6) {
        // 10Hz seems to work on 6
        ubx_cfg_set_rate(gps_port, (uint16_t)100);
    } else {
        // 5Hz
        ubx_cfg_set_rate(gps_port, (uint16_t)200);        
    }
}

//! Parse incoming data while paused
static void ubx_cfg_pause_parse(uintptr_t gps_port, uint32_t delay_ticks)
{
    struct GPS_RX_STATS gpsRxStats;
    GPSPositionData     gpsPosition;

    uint8_t c;
    uint32_t enterTime = PIOS_Thread_Systime();
    while ((PIOS_Thread_Systime() - enterTime) < delay_ticks)
    {
        int32_t received = PIOS_COM_ReceiveBuffer(gps_port, &c, 1, 1);
        if (received > 0)
            parse_ubx_stream (c, gps_rx_buffer, &gpsPosition, &gpsRxStats);
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
    PIOS_COM_SendChar(gps_port, UBLOX_SYNC1);
    PIOS_COM_SendChar(gps_port, UBLOX_SYNC2);
    PIOS_COM_SendBuffer(gps_port, dat, len);
    PIOS_COM_SendChar(gps_port, ubloxTxCK_A);
    PIOS_COM_SendChar(gps_port, ubloxTxCK_B);

    ubx_cfg_pause_parse(gps_port, UBLOX_WAIT_MS);
}

/**
 * Completely configure a UBX GPS with the messages we expect
 * in NAV5 mode at the appropriate rate.
 */
void ubx_cfg_send_configuration(uintptr_t gps_port, char *buffer,
        ModuleSettingsGPSConstellationOptions constellation,
        ModuleSettingsGPSSBASConstellationOptions sbas_const,
        ModuleSettingsGPSDynamicsModeOptions dyn_mode)
{
    gps_rx_buffer = buffer;

    // Enable this to clear GPS. Not done by default
    // because we don't want to keep writing to the
    // persistent storage
    if (false) ubx_cfg_clear_cfg(gps_port);

    ubx_cfg_set_timepulse(gps_port);

    UBloxInfoData ublox;

    // Poll the version number and parse some data
    uint32_t i = 0;
    do {
        ubx_cfg_poll_version(gps_port);
        ubx_cfg_pause_parse(gps_port, UBLOX_WAIT_MS);
        UBloxInfoGet(&ublox);
    } while (ublox.swVersion == 0 && i++ < 10);

    ubx_cfg_enable_message(gps_port, UBLOX_NAV_CLASS, UBLOX_NAV_VELNED, 1);    // NAV-VELNED
    ubx_cfg_enable_message(gps_port, UBLOX_NAV_CLASS, UBLOX_NAV_POSLLH, 1);    // NAV-POSLLH
    ubx_cfg_enable_message(gps_port, UBLOX_NAV_CLASS, UBLOX_NAV_SOL, 1);       // NAV-SOL
    ubx_cfg_enable_message(gps_port, UBLOX_NAV_CLASS, UBLOX_NAV_TIMEUTC, 5);   // NAV-TIMEUTC
    ubx_cfg_enable_message(gps_port, UBLOX_NAV_CLASS, UBLOX_NAV_DOP, 1);       // NAV-DOP
    ubx_cfg_enable_message(gps_port, UBLOX_NAV_CLASS, UBLOX_NAV_SVINFO, 5);    // NAV-SVINFO

    ubx_cfg_set_mode(gps_port, dyn_mode);

    // Hardcoded version. The poll version method should fetch the
    // data but we need to link to that.
    if (ublox.swVersion > 0)
        ubx_cfg_version_specific(gps_port, floorf(ublox.swVersion),
                constellation, sbas_const);
    else
        ubx_cfg_version_specific(gps_port, 6,
                constellation, sbas_const);
}

//! Make sure the GPS is set to the same baudrate as the port
void ubx_cfg_set_baudrate(uintptr_t gps_port, ModuleSettingsGPSSpeedOptions baud_rate)
{
    // UBX,41 msg
    // 1 - portID
    // 0007 - input protocol (all)
    // 0001 - output protocol (ubx only)
    // 0 - no attempt to autobaud
    // number - baudrate
    // *XX - checksum
    const char * msg_2400 = "$PUBX,41,1,0007,0001,2400,0*1B\r\n";
    const char * msg_4800 = "$PUBX,41,1,0007,0001,4800,0*11\r\n";
    const char * msg_9600 = "$PUBX,41,1,0007,0001,9600,0*12\r\n";
    const char * msg_19200 = "$PUBX,41,1,0007,0001,19200,0*27\r\n";
    const char * msg_38400 = "$PUBX,41,1,0007,0001,38400,0*22\r\n";
    const char * msg_57600 = "$PUBX,41,1,0007,0001,57600,0*29\r\n";
    const char * msg_115200 = "$PUBX,41,1,0007,0001,115200,0*1A\r\n";
    const char * msg_230400 = "$PUBX,41,1,0007,0001,230400,0*18\r\n";

    const char *msg;
    uint32_t baud;
    switch (baud_rate) {
    case MODULESETTINGS_GPSSPEED_2400:
        msg = msg_2400;
        baud = 2400;
        break;
    case MODULESETTINGS_GPSSPEED_4800:
        msg = msg_4800;
        baud = 4800;
        break;
    case MODULESETTINGS_GPSSPEED_9600:
        msg = msg_9600;
        baud = 9600;
        break;
    case MODULESETTINGS_GPSSPEED_19200:
        msg = msg_19200;
        baud = 19200;
        break;
    case MODULESETTINGS_GPSSPEED_38400:
        msg = msg_38400;
        baud = 38400;
        break;
    default:
    case MODULESETTINGS_GPSSPEED_57600:
        msg = msg_57600;
        baud = 57600;
        break;
    case MODULESETTINGS_GPSSPEED_115200:
        msg = msg_115200;
        baud = 115200;
        break;
    case MODULESETTINGS_GPSSPEED_230400:
        msg = msg_230400;
        baud = 230400;
        break;
    }

    // Attempt to set baud rate to desired value from a number of
    // common rates. So this configures the physical baudrate and
    // tries to send the configuration string to the GPS.
    const uint32_t baud_rates[] = {2400, 4800, 9600, 19200, 38400, 57600, 115200, 230400};
    for (uint32_t i = 0; i < NELEMENTS(baud_rates); i++) {
        PIOS_COM_ChangeBaud(gps_port, baud_rates[i]);
        PIOS_Thread_Sleep(UBLOX_WAIT_MS);

        // Send the baud rate change message
        PIOS_COM_SendString(gps_port, msg);

        // Wait until the message has been fully transmitted including all start+stop bits
        // 34 bytes * 10bits/byte = 340 bits
        // At 2400bps, that's (340 / 2400) = 142ms
        // add some margin and we end up with 200ms
        PIOS_Thread_Sleep(200);
    }

    // Set to proper baud rate
    PIOS_COM_ChangeBaud(gps_port, baud);
}

#endif /* PIOS_GPS_MINIMAL */

/**
 * @}
 * @}
 */
