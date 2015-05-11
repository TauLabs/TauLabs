/**
******************************************************************************
* @addtogroup PIOS PIOS Core hardware abstraction layer
* @{
* @addtogroup PIOS_RFM22B Radio Functions
* @brief PIOS OpenLRS interface for for the RFM22B radio
* @{
*
* @file       pios_openlrs_priv.h
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
* @brief      Implements an OpenLRS driver for the RFM22B
* @see        The GNU Public License (GPL) Version 3
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

#ifndef PIOS_OPENLRS_PRIV_H
#define PIOS_OPENLRS_PRIV_H

#include "pios_openlrs.h"

#define OPENLRSNG_VERSION 0x0382
 
#define MAXHOPS      24
#define OPENLRS_PPM_NUM_CHANNELS 16

// Factory setting values, modify via the CLI

//####### RADIOLINK RF POWER (beacon is always 100/13/1.3mW) #######
// 7 == 100mW (or 1000mW with M3)
// 6 == 50mW (use this when using booster amp), (800mW with M3)
// 5 == 25mW
// 4 == 13mW
// 3 == 6mW
// 2 == 3mW
// 1 == 1.6mW
// 0 == 1.3mW
#define DEFAULT_RF_POWER 7

#define DEFAULT_CHANNEL_SPACING 5 // 50kHz
#define DEFAULT_HOPLIST 22,10,19,34,49,41
#define DEFAULT_RF_MAGIC 0xDEADFEED

//  0 -- 4800bps, best range
//  1 -- 9600bps, medium range
//  2 -- 19200bps, medium range
#define DEFAULT_DATARATE 2

// BIND_DATA flag masks
#define TELEMETRY_OFF       0x00
#define TELEMETRY_PASSTHRU  0x08
#define TELEMETRY_FRSKY     0x10 // covers smartport if used with &
#define TELEMETRY_SMARTPORT 0x18
#define TELEMETRY_MASK      0x18
#define CHANNELS_4_4        0x01
#define CHANNELS_8          0x02
#define CHANNELS_8_4        0x03
#define CHANNELS_12         0x04
#define CHANNELS_12_4       0x05
#define CHANNELS_16         0x06
#define DIVERSITY_ENABLED   0x80
#define DEFAULT_FLAGS       (CHANNELS_8 | TELEMETRY_PASSTHRU)

// helper macro for European PMR channels
#define EU_PMR_CH(x) (445993750L + 12500L * (x)) // valid for ch1-ch8

// helper macro for US FRS channels 1-7
#define US_FRS_CH(x) (462537500L + 25000L * (x)) // valid for ch1-ch7

#define DEFAULT_BEACON_FREQUENCY 0 // disable beacon
#define DEFAULT_BEACON_DEADTIME 30 // time to wait until go into beacon mode (30s)
#define DEFAULT_BEACON_INTERVAL 10 // interval between beacon transmits (10s)

#define BINDING_POWER     0x06 // not lowest since may result fail with RFM23BP

#define TELEMETRY_PACKETSIZE 9

#define BIND_MAGIC (0xDEC1BE15 + (OPENLRSNG_VERSION & 0xfff0))
#define BINDING_VERSION ((OPENLRSNG_VERSION & 0x0ff0)>>4)

// HW frequency limits
#if (RFMTYPE == 868)
#  define MIN_RFM_FREQUENCY 848000000
#  define MAX_RFM_FREQUENCY 888000000
#  define DEFAULT_CARRIER_FREQUENCY 868000000  // Hz  (ch 0)
#  define BINDING_FREQUENCY 868000000 // Hz
#elif (RFMTYPE == 915)
#  define MIN_RFM_FREQUENCY 895000000
#  define MAX_RFM_FREQUENCY 935000000
#  define DEFAULT_CARRIER_FREQUENCY 915000000  // Hz  (ch 0)
#  define BINDING_FREQUENCY 915000000 // Hz
#else
#  define MIN_RFM_FREQUENCY 413000000
#  define MAX_RFM_FREQUENCY 463000000
#  define DEFAULT_CARRIER_FREQUENCY 435000000  // Hz  (ch 0)
#  define BINDING_FREQUENCY 435000000 // Hz
#endif

#define RFM22_DEVICE_TYPE                         0x00  // R
#define RFM22_DT_MASK                             0x1F

struct bind_data {
  uint8_t version;
  uint32_t serial_baudrate;
  uint32_t rf_frequency;
  uint32_t rf_magic;
  uint8_t rf_power;
  uint8_t rf_channel_spacing;
  uint8_t hopchannel[MAXHOPS];
  uint8_t modem_params;
  uint8_t flags;
} __attribute__((packed));

enum RF_MODE {
  Available, Transmit, Receive, Transmitted, Received,
};

enum pios_openlrs_dev_magic {
  PIOS_OPENLRS_DEV_MAGIC = 0x18c97ab6,
};

struct pios_openlrs_dev {
  enum pios_openlrs_dev_magic magic;
  struct pios_openlrs_cfg cfg;

  // The SPI bus information
  uint32_t spi_id;
  uint32_t slave_num;

  // The task handle
  struct pios_thread *taskHandle;

  // The COM callback functions.
  pios_com_callback rx_in_cb;
  uint32_t rx_in_context;
  pios_com_callback tx_out_cb;
  uint32_t tx_out_context;

  // The event queue handle
  struct pios_semaphore *sema_isr;

  // The PPM buffer
  int16_t ppm[OPENLRS_PPM_NUM_CHANNELS];

  // RFM22B RCVR interface
  uintptr_t openlrs_rcvr_id;

  // Flag to indicate if link every acquired
  bool link_acquired;

  // Active bound information data
  struct bind_data bind_data;

  // Beacon settings
  uint32_t beacon_frequency;
  uint8_t beacon_delay;
  uint8_t beacon_period;

  enum RF_MODE rf_mode;
  uint32_t rf_channel;

  uint8_t it_status1;
  uint8_t it_status2;

  uint8_t rx_buf[64];
  uint8_t tx_buf[9];

  // Variables from OpenLRS for radio control
  uint8_t hopcount;
  uint32_t lastPacketTimeUs;
  uint32_t numberOfLostPackets;
  uint16_t lastAFCCvalue;
  uint16_t linkQuality;
  uint32_t lastRSSITimeUs;
  uint8_t lastRSSIvalue;
  bool willhop;
  uint32_t nextBeaconTimeMs;
  uint32_t linkLossTimeMs;
  bool failsafeActive;
  uint32_t failsafeDelay;
  uint32_t beacon_rssi_avg;
};

bool PIOS_OpenLRS_EXT_Int(void);

#endif /* PIOS_OPENLRS_PRIV_H */
/**
 * @}
 * @}
 */
