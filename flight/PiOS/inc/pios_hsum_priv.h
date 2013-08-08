/**
 ******************************************************************************
 * @file       pios_hsum.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HSUM Graupner HoTT receiver functions
 * @{
 * @brief Graupner HoTT receiver functions for SUMD/H
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

#ifndef PIOS_HSUM_PRIV_H
#define PIOS_HSUM_PRIV_H

#include <pios.h>
#include <pios_usart_priv.h>

/*
 * Currently known Graupner HoTT serial port settings:
 *  115200bps serial stream, 8 bits, no parity, 1 stop bit
 *  size of each frame: 11..37 bytes
 *  data resolution: 14 bit
 *  frame period: 11ms or 22ms
 *
 * Currently known SUMD/SUMH frame structure:
 * Section          Byte_Number        Byte_Name      Byte_Value Remark
 * Header           0                  Vendor_ID      0xA8       Graupner
 * Header           1                  Status         0x00       valid and live SUMH data frame
 *                                                    0x01       valid and live SUMD data frame
 *                                                    0x81       valid SUMD/H data frame with
 *                                                               transmitter in fail safe condition
 *                                                    others     invalid frame
 * Header           2                  N_Channels     0x02..0x20 number of transmitted channels
 * Data             n*2+1              Channel n MSB  0x00..0xff High Byte of channel n data
 * Data             n*2+2              Channel n LSB  0x00..0xff Low Byte of channel n data
 * SUMD_CRC         (N_Channels+1)*2+1 CRC High Byte  0x00..0xff High Byte of 16 Bit CRC
 * SUMD_CRC         (N_Channels+1)*2+2 CRC Low Byte   0x00..0xff Low Byte of 16 Bit CRC
 * SUMH_Telemetry   (N_Channels+1)*2+1 Telemetry_Req  0x00..0xff 0x00 no telemetry request
 * SUMH_CRC         (N_Channels+1)*2+2 CRC Byte       0x00..0xff Low Byte of all added data bytes


 Channel Data Interpretation
 Stick Positon    Channel Data Remark
 ext. low (-150%) 0x1c20       900µs
 low (-100%)      0x2260       1100µs
 netral (0%)      0x2ee0       1500µs
 high (100%)      0x3b60       1900µs
 ext. high(150%)  0x41a0       2100µs
 
 Channel Mapping (not sure)
 1 Pitch
 2 Aileron
 3 Elevator
 4 Yaw
 5 Aux/Gyro on MX-12
 6 ESC
 7 Aux/Gyr
 */

/* HSUM frame size and contents definitions */
#define HSUM_MAX_CHANNELS_PER_FRAME 32
#define HSUM_MAX_FRAME_LENGTH (HSUM_MAX_CHANNELS_PER_FRAME*2+5)
#define HSUM_H

#define HSUM_GRAUPNER_ID 0xA8
#define HSUM_STATUS_LIVING_SUMH 0x00
#define HSUM_STATUS_LIVING_SUMD 0x01
#define HSUM_STATUS_FAILSAFE 0x81

/* HSUM protocol variations */
enum pios_hsum_proto {
	PIOS_HSUM_PROTO_SUMD,
	PIOS_HSUM_PROTO_SUMH,
};

/* HSUM receiver instance configuration */
extern const struct pios_rcvr_driver pios_hsum_rcvr_driver;

extern int32_t PIOS_HSUM_Init(uintptr_t *hsum_id,
			     const struct pios_com_driver *driver,
			     uintptr_t lower_id,
			     enum pios_hsum_proto proto);

#endif /* PIOS_HSUM_PRIV_H */

/**
 * @}
 * @}
 */
