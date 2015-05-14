/**
******************************************************************************
* @addtogroup PIOS PIOS Core hardware abstraction layer
* @{
* @addtogroup	PIOS_RFM22B Radio Functions
* @brief PIOS OpenLRS interface for for the RFM22B radio
* @{
*
* @file       pios_openlrs.c
* @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
* @brief      Implements an OpenLRS driver for the RFM22B
* @see	      The GNU Public License (GPL) Version 3
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

#include "pios.h"

#ifdef PIOS_INCLUDE_OPENLRS

#include <pios_thread.h>
#include <pios_spi_priv.h>
#include <pios_openlrs_priv.h>
#include <pios_openlrs_rcvr_priv.h>
#include <taskmonitor.h>
#include <taskinfo.h>

#include "openlrs.h"
#include "flightstatus.h"
#include "flightbatterystate.h"

#include "pios_rfm22b_regs.h"

#define STACK_SIZE_BYTES                 800
#define TASK_PRIORITY                    PIOS_THREAD_PRIO_NORMAL


static void rx_reset(struct pios_openlrs_dev *openlrs_dev);
static void rfmSetCarrierFrequency(struct pios_openlrs_dev *openlrs_dev, uint32_t f);
static uint8_t rfmGetRSSI(struct pios_openlrs_dev *openlrs_dev);
static void to_rx_mode(struct pios_openlrs_dev *openlrs_dev);
static void tx_packet(struct pios_openlrs_dev *openlrs_dev, uint8_t* pkt, uint8_t size);

static struct pios_openlrs_dev * pios_openlrs_alloc();
static bool pios_openlrs_validate(struct pios_openlrs_dev *openlrs_dev);

// SPI read/write functions
static void rfm22_assertCs(struct pios_openlrs_dev *openlrs_dev);
static void rfm22_deassertCs(struct pios_openlrs_dev *openlrs_dev);
static void rfm22_claimBus(struct pios_openlrs_dev *openlrs_dev);
static void rfm22_releaseBus(struct pios_openlrs_dev *openlrs_dev);
static void rfm22_write_claim(struct pios_openlrs_dev *openlrs_dev,
			      uint8_t addr, uint8_t data);
static void rfm22_write(struct pios_openlrs_dev *openlrs_dev, uint8_t addr,
			uint8_t data);
static uint8_t rfm22_read_claim(struct pios_openlrs_dev *openlrs_dev,
			  uint8_t addr);
static uint8_t rfm22_read(struct pios_openlrs_dev *openlrs_dev,
			  uint8_t addr);

// Private constants
const struct rfm22_modem_regs {
  uint32_t bps;
  uint8_t  r_1c, r_1d, r_1e, r_20, r_21, r_22, r_23, r_24, r_25, r_2a, r_6e, r_6f, r_70, r_71, r_72;
} modem_params[] = {
  { 4800, 0x1a, 0x40, 0x0a, 0xa1, 0x20, 0x4e, 0xa5, 0x00, 0x1b, 0x1e, 0x27, 0x52, 0x2c, 0x23, 0x30 }, // 50000 0x00
  { 9600, 0x05, 0x40, 0x0a, 0xa1, 0x20, 0x4e, 0xa5, 0x00, 0x20, 0x24, 0x4e, 0xa5, 0x2c, 0x23, 0x30 }, // 25000 0x00
  { 19200, 0x06, 0x40, 0x0a, 0xd0, 0x00, 0x9d, 0x49, 0x00, 0x7b, 0x28, 0x9d, 0x49, 0x2c, 0x23, 0x30 }, // 25000 0x01
  { 57600, 0x05, 0x40, 0x0a, 0x45, 0x01, 0xd7, 0xdc, 0x03, 0xb8, 0x1e, 0x0e, 0xbf, 0x00, 0x23, 0x2e },
  { 125000, 0x8a, 0x40, 0x0a, 0x60, 0x01, 0x55, 0x55, 0x02, 0xad, 0x1e, 0x20, 0x00, 0x00, 0x23, 0xc8 },
};

const static uint8_t pktsizes[8] = { 0, 7, 11, 12, 16, 17, 21, 0 };

static const uint8_t OUT_FF[64] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

const uint8_t default_hop_list[] = {DEFAULT_HOPLIST};

const struct rfm22_modem_regs bind_params =
{ 9600, 0x05, 0x40, 0x0a, 0xa1, 0x20, 0x4e, 0xa5, 0x00, 0x20, 0x24, 0x4e, 0xa5, 0x2c, 0x23, 0x30 };

/*****************************************************************************
* OpenLRS data formatting utilities
*****************************************************************************/

static uint8_t getPacketSize(struct bind_data *bd)
{
	return pktsizes[(bd->flags & 0x07)];
}

static uint32_t getInterval(struct bind_data *bd)
{
	uint32_t ret;
	// Sending a x byte packet on bps y takes about (emperical)
	// usec = (x + 15) * 8200000 / baudrate
#define BYTES_AT_BAUD_TO_USEC(bytes, bps, div) ((uint32_t)((bytes) + (div?20:15)) * 8200000L / (uint32_t)(bps))

	ret = (BYTES_AT_BAUD_TO_USEC(getPacketSize(bd), modem_params[bd->modem_params].bps, bd->flags&DIVERSITY_ENABLED) + 2000);

	if (bd->flags & TELEMETRY_MASK) {
		ret += (BYTES_AT_BAUD_TO_USEC(TELEMETRY_PACKETSIZE, modem_params[bd->modem_params].bps, bd->flags&DIVERSITY_ENABLED) + 1000);
	}

	// round up to ms
	ret = ((ret + 999) / 1000) * 1000;

  // enable following to limit packet rate to 50Hz at most
#ifdef LIMIT_RATE_TO_50HZ
	if (ret < 20000) {
		ret = 20000;
	}
#endif

	return ret;
}

static void unpackChannels(uint8_t config, int16_t PPM[], uint8_t *p)
{
	uint8_t i;
	for (i=0; i<=(config/2); i++) { // 4ch packed in 5 bytes
		PPM[0] = (((uint16_t)p[4] & 0x03) << 8) + p[0];
		PPM[1] = (((uint16_t)p[4] & 0x0c) << 6) + p[1];
		PPM[2] = (((uint16_t)p[4] & 0x30) << 4) + p[2];
		PPM[3] = (((uint16_t)p[4] & 0xc0) << 2) + p[3];
		p+=5;
		PPM+=4;
	}
	if (config & 1) { // 4ch packed in 1 byte;
		PPM[0] = (((uint16_t)p[0] >> 6) & 3) * 333 + 12;
		PPM[1] = (((uint16_t)p[0] >> 4) & 3) * 333 + 12;
		PPM[2] = (((uint16_t)p[0] >> 2) & 3) * 333 + 12;
		PPM[3] = (((uint16_t)p[0] >> 0) & 3) * 333 + 12;
	}
}

//! Apply the OpenLRS rescaling to the channels
static void rescaleChannels(int16_t PPM[])
{
	for (uint32_t i = 0; i < OPENLRS_PPM_NUM_CHANNELS; i++) {
		int16_t x = PPM[i];
		int16_t ret;

		if (x < 12) {
			ret = 808 + x * 16;
		} else if (x < 1012) {
			ret = x + 988;
		} else if (x < 1024) {
			ret = 2000 + (x - 1011) * 16;
		} else {
			ret = 2192;
		}

		PPM[i] = ret;
	}
}

static uint8_t countSetBits(uint16_t x)
{
	x  = x - ((x >> 1) & 0x5555);
	x  = (x & 0x3333) + ((x >> 2) & 0x3333);
	x  = x + (x >> 4);
	x &= 0x0F0F;
	return (x * 0x0101) >> 8;
}

static uint32_t micros()
{
	return PIOS_DELAY_GetuS();
}

static uint32_t millis()
{
	return PIOS_Thread_Systime();
}
static void delay(uint32_t time)
{
	// TODO: confirm this is in ms
	PIOS_Thread_Sleep(time);
}

/*****************************************************************************
* OpenLRS hardware access
*****************************************************************************/

#define NOP() __asm__ __volatile__("nop")

#define RF22B_PWRSTATE_POWERDOWN    0x00
#define RF22B_PWRSTATE_READY	    RFM22_opfc1_xton
#define RF22B_PWRSTATE_RX	        (RFM22_opfc1_rxon | RFM22_opfc1_xton)
#define RF22B_PWRSTATE_TX	        (RFM22_opfc1_txon | RFM22_opfc1_xton)

#define RF22B_PACKET_SENT_INTERRUPT          RFM22_ie1_enpksent
#define RF22B_RX_PACKET_RECEIVED_IRQ         RFM22_ie1_enpkvalid

static void rfmSetChannel(struct pios_openlrs_dev *openlrs_dev, uint8_t ch)
{
	DEBUG_PRINTF(3,"rfmSetChannel %d\r\n", ch);
	uint8_t magicLSB = (openlrs_dev->bind_data.rf_magic & 0xff) ^ ch;
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_frequency_hopping_channel_select, openlrs_dev->bind_data.hopchannel[ch]);
	rfm22_write(openlrs_dev, RFM22_transmit_header3 + 3, magicLSB);
	rfm22_write(openlrs_dev, RFM22_check_header3 + 3, magicLSB);
	rfm22_releaseBus(openlrs_dev);
}

static uint8_t rfmGetRSSI(struct pios_openlrs_dev *openlrs_dev)
{
	return rfm22_read_claim(openlrs_dev, 0x26);
}

static uint16_t rfmGetAFCC(struct pios_openlrs_dev *openlrs_dev)
{
	return (((uint16_t)rfm22_read_claim(openlrs_dev, 0x2B) << 2) | ((uint16_t)rfm22_read_claim(openlrs_dev, 0x2C) >> 6));
}

static void setModemRegs(struct pios_openlrs_dev *openlrs_dev, const struct rfm22_modem_regs* r)
{
	DEBUG_PRINTF(3,"setModemRegs\r\n");
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_if_filter_bandwidth, r->r_1c);
	rfm22_write(openlrs_dev, RFM22_afc_loop_gearshift_override, r->r_1d);
	rfm22_write(openlrs_dev, RFM22_afc_timing_control, r->r_1e);
	rfm22_write(openlrs_dev, RFM22_clk_recovery_oversampling_ratio, r->r_20);
	rfm22_write(openlrs_dev, RFM22_clk_recovery_offset2, r->r_21);
	rfm22_write(openlrs_dev, RFM22_clk_recovery_offset1, r->r_22);
	rfm22_write(openlrs_dev, RFM22_clk_recovery_offset0, r->r_23);
	rfm22_write(openlrs_dev, RFM22_clk_recovery_timing_loop_gain1, r->r_24);
	rfm22_write(openlrs_dev, RFM22_clk_recovery_timing_loop_gain0, r->r_25);
	rfm22_write(openlrs_dev, RFM22_afc_limiter, r->r_2a);
	rfm22_write(openlrs_dev, RFM22_tx_data_rate1, r->r_6e);
	rfm22_write(openlrs_dev, RFM22_tx_data_rate0, r->r_6f);
	rfm22_write(openlrs_dev, RFM22_modulation_mode_control1, r->r_70);
	rfm22_write(openlrs_dev, RFM22_modulation_mode_control2, r->r_71);
	rfm22_write(openlrs_dev, RFM22_frequency_deviation, r->r_72);
	rfm22_releaseBus(openlrs_dev);
}

static void rfmSetCarrierFrequency(struct pios_openlrs_dev *openlrs_dev, uint32_t f)
{
	DEBUG_PRINTF(3,"rfmSetCarrierFrequency %d\r\n", f);
	uint16_t fb, fc, hbsel;
	if (f < 480000000) {
		hbsel = 0;
		fb = f / 10000000 - 24;
		fc = (f - (fb + 24) * 10000000) * 4 / 625;
	} else {
		hbsel = 1;
		fb = f / 20000000 - 24;
		fc = (f - (fb + 24) * 20000000) * 2 / 625;
	}
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_frequency_band_select, RFM22_fbs_sbse + (hbsel ? RFM22_fbs_hbsel : 0) + (fb & RFM22_fb_mask));
	rfm22_write(openlrs_dev, RFM22_nominal_carrier_frequency1, (fc >> 8));
	rfm22_write(openlrs_dev, RFM22_nominal_carrier_frequency0, (fc & 0xff));
	rfm22_releaseBus(openlrs_dev);
}

static void init_rfm(struct pios_openlrs_dev *openlrs_dev, uint8_t isbind)
{
	DEBUG_PRINTF(2,"init_rfm %d\r\n", isbind);

	if (!isbind) {
		DEBUG_PRINTF(2, "Binding settings:\r\n");
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  version: %d\r\n", openlrs_dev->bind_data.version);
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  serial_baudrate: %d\r\n", openlrs_dev->bind_data.serial_baudrate);
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  rf_frequency: %d\r\n", openlrs_dev->bind_data.rf_frequency);
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  rf_power: %d\r\n", openlrs_dev->bind_data.rf_power);
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  rf_channel_spacing: %d\r\n", openlrs_dev->bind_data.rf_channel_spacing);
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  modem_params: %d\r\n", openlrs_dev->bind_data.modem_params);
		PIOS_Thread_Sleep(10);
		DEBUG_PRINTF(2, "  flags: %d\r\n", openlrs_dev->bind_data.flags);
		PIOS_Thread_Sleep(10);
	}

	rfm22_claimBus(openlrs_dev);
	openlrs_dev->it_status1 = rfm22_read(openlrs_dev, RFM22_interrupt_status1);   // read status, clear interrupt
	openlrs_dev->it_status2 = rfm22_read(openlrs_dev, RFM22_interrupt_status2);
	rfm22_write(openlrs_dev, RFM22_interrupt_enable2, 0x00);    // disable interrupts
	rfm22_write(openlrs_dev, RFM22_op_and_func_ctrl1, RF22B_PWRSTATE_READY); // disable lbd, wakeup timer, use internal 32768,xton = 1; in ready mode
	rfm22_write(openlrs_dev, RFM22_xtal_osc_load_cap, 0x7f);   // c = 12.5p
	rfm22_write(openlrs_dev, RFM22_cpu_output_clk, 0x05);
	switch (openlrs_dev->cfg.gpio_direction) {
	case GPIO0_TX_GPIO1_RX:
		rfm22_write(openlrs_dev, RFM22_gpio0_config, RFM22_gpio0_config_txstate);    // gpio0 TX State
		rfm22_write(openlrs_dev, RFM22_gpio1_config, RFM22_gpio1_config_rxstate);    // gpio1 RX State
		break;
	case GPIO0_RX_GPIO1_TX:
		rfm22_write(openlrs_dev, RFM22_gpio0_config, RFM22_gpio0_config_rxstate);    // gpio0 RX State
		rfm22_write(openlrs_dev, RFM22_gpio1_config, RFM22_gpio1_config_txstate);    // gpio1 TX State
		break;
	}
	rfm22_write(openlrs_dev, RFM22_gpio2_config, 0xfd);    // gpio 2 VDD
	rfm22_write(openlrs_dev, RFM22_io_port_config, RFM22_io_port_default);    // gpio    0, 1,2 NO OTHER FUNCTION.
	rfm22_releaseBus(openlrs_dev);

	if (isbind) {
		setModemRegs(openlrs_dev, &bind_params);
	} else {
		setModemRegs(openlrs_dev, &modem_params[openlrs_dev->bind_data.modem_params]);
	}

	// Packet settings
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_data_access_control, 0x8c);    // enable packet handler, msb first, enable crc,
	rfm22_write(openlrs_dev, RFM22_header_control1, 0x0f);    // no broadcast, check header bytes 3,2,1,0
	rfm22_write(openlrs_dev, RFM22_header_control2, 0x42);    // 4 byte header, 2 byte synch, variable pkt size
	rfm22_write(openlrs_dev, RFM22_preamble_length, (openlrs_dev->bind_data.flags & DIVERSITY_ENABLED)?0x14:0x0a);    // 40 bit preamble, 80 with diversity
	rfm22_write(openlrs_dev, RFM22_preamble_detection_ctrl1, 0x2a);    // preath = 5 (20bits), rssioff = 2
	rfm22_write(openlrs_dev, RFM22_sync_word3, 0x2d);    // synchronize word 3
	rfm22_write(openlrs_dev, RFM22_sync_word2, 0xd4);    // synchronize word 2
	rfm22_write(openlrs_dev, RFM22_sync_word1, 0x00);    // synch word 1 (not used)
	rfm22_write(openlrs_dev, RFM22_sync_word0, 0x00);    // synch word 0 (not used)

	uint32_t magic = isbind ? BIND_MAGIC : openlrs_dev->bind_data.rf_magic;
	for (uint8_t i = 0; i < 4; i++) {
		rfm22_write(openlrs_dev, RFM22_transmit_header3 + i, (magic >> 24) & 0xff);   // tx header
		rfm22_write(openlrs_dev, RFM22_check_header3 + i, (magic >> 24) & 0xff);   // rx header
		magic = magic << 8; // advance to next byte
	}

	rfm22_write(openlrs_dev, RFM22_header_enable3, 0xff);    // all the bit to be checked
	rfm22_write(openlrs_dev, RFM22_header_enable2, 0xff);    // all the bit to be checked
	rfm22_write(openlrs_dev, RFM22_header_enable1, 0xff);    // all the bit to be checked
	rfm22_write(openlrs_dev, RFM22_header_enable0, 0xff);    // all the bit to be checked

	if (isbind) {
		rfm22_write(openlrs_dev, RFM22_tx_power, BINDING_POWER);
	} else {
		rfm22_write(openlrs_dev, RFM22_tx_power, openlrs_dev->bind_data.rf_power);
	}

	rfm22_write(openlrs_dev, RFM22_frequency_hopping_channel_select, 0);
	rfm22_write(openlrs_dev, RFM22_frequency_hopping_step_size, openlrs_dev->bind_data.rf_channel_spacing);   // channel spacing

	rfm22_write(openlrs_dev, RFM22_frequency_offset1, 0x00);
	rfm22_write(openlrs_dev, RFM22_frequency_offset2, 0x00);    // no offset

	rfm22_releaseBus(openlrs_dev);

	rfmSetCarrierFrequency(openlrs_dev, isbind ? BINDING_FREQUENCY : openlrs_dev->bind_data.rf_frequency);
}

static void to_rx_mode(struct pios_openlrs_dev *openlrs_dev)
{
	DEBUG_PRINTF(3,"to_rx_mode\r\n");
	rfm22_claimBus(openlrs_dev);
	openlrs_dev->it_status1 = rfm22_read(openlrs_dev, RFM22_interrupt_status1);
	openlrs_dev->it_status2 = rfm22_read(openlrs_dev, RFM22_interrupt_status2);
	rfm22_write(openlrs_dev, RFM22_op_and_func_ctrl1, RF22B_PWRSTATE_READY);
	rfm22_releaseBus(openlrs_dev);
	delay(10);
	rx_reset(openlrs_dev);
	NOP();
}

static void clearFIFO(struct pios_openlrs_dev *openlrs_dev)
{
	DEBUG_PRINTF(3,"clearFIFO\r\n");
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_op_and_func_ctrl2, 0x03);
	rfm22_write(openlrs_dev, RFM22_op_and_func_ctrl2, 0x00);
	rfm22_releaseBus(openlrs_dev);
}

static void rx_reset(struct pios_openlrs_dev *openlrs_dev)
{
	DEBUG_PRINTF(3,"rx_reset\r\n");
	rfm22_write_claim(openlrs_dev, RFM22_op_and_func_ctrl1, RF22B_PWRSTATE_READY);
	rfm22_write_claim(openlrs_dev, RFM22_rx_fifo_control, 36);	 // threshold for rx almost full, interrupt when 1 byte received
	clearFIFO(openlrs_dev);
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_op_and_func_ctrl1, RF22B_PWRSTATE_RX);   // to rx mode
	rfm22_write(openlrs_dev, RFM22_interrupt_enable1, RF22B_RX_PACKET_RECEIVED_IRQ);
	openlrs_dev->it_status1 = rfm22_read(openlrs_dev, RFM22_interrupt_status1);   //read the Interrupt Status1 register
	openlrs_dev->it_status2 = rfm22_read(openlrs_dev, RFM22_interrupt_status2);
	rfm22_releaseBus(openlrs_dev);
}

// TODO: move into dev structure
uint32_t tx_start = 0;

static void tx_packet_async(struct pios_openlrs_dev *openlrs_dev, uint8_t* pkt, uint8_t size)
{
	rfm22_claimBus(openlrs_dev);
	rfm22_write(openlrs_dev, RFM22_transmit_packet_length, size);   // total tx size

	for (uint8_t i = 0; i < size; i++) {
		rfm22_write(openlrs_dev, RFM22_fifo_access, pkt[i]);
	}

	rfm22_write(openlrs_dev, RFM22_interrupt_enable1, RF22B_PACKET_SENT_INTERRUPT);
	openlrs_dev->it_status1 = rfm22_read(openlrs_dev, RFM22_interrupt_status1);	  //read the Interrupt Status1 register
	openlrs_dev->it_status2 = rfm22_read(openlrs_dev, RFM22_interrupt_status2);
	tx_start = micros();
	rfm22_write(openlrs_dev, RFM22_op_and_func_ctrl1, RF22B_PWRSTATE_TX);	// to tx mode
	rfm22_releaseBus(openlrs_dev);

	openlrs_dev->rf_mode = Transmit;
}

static void tx_packet(struct pios_openlrs_dev *openlrs_dev, uint8_t* pkt, uint8_t size)
{
	tx_packet_async(openlrs_dev, pkt, size);
	PIOS_Semaphore_Take(openlrs_dev->sema_isr, 25);

#if defined(PIOS_INCLUDE_WDG) && defined(PIOS_WDG_RFM22B)
		// Update the watchdog timer
		PIOS_WDG_UpdateFlag(PIOS_WDG_RFM22B);
#endif /* PIOS_WDG_RFM22B */

	if (openlrs_dev->rf_mode == Transmit) {
		DEBUG_PRINTF(2,"tx_packet timeout\r\n");
		init_rfm(openlrs_dev, false); // reset modem
	}
}

static void beacon_tone(struct pios_openlrs_dev *openlrs_dev, int16_t hz, int16_t len) //duration is now in half seconds.
{
	DEBUG_PRINTF(2,"beacon_tone: %d %d\r\n", hz, len*2);
	int16_t d = 500000 / hz; // better resolution

#if defined(PIOS_LED_LINK)
	PIOS_LED_On(PIOS_LED_LINK);
#endif /* PIOS_LED_LINK */

	if (d < 1) {
		d = 1;
	}

	rfm22_claimBus(openlrs_dev);

	GPIO_TypeDef* gpio = openlrs_dev->cfg.spi_cfg->mosi.gpio;
	uint16_t pin_source = openlrs_dev->cfg.spi_cfg->mosi.init.GPIO_Pin;
	uint8_t remap = openlrs_dev->cfg.spi_cfg->remap;

	GPIO_InitTypeDef init = {
		.GPIO_Speed = GPIO_Speed_50MHz,
		.GPIO_Mode  = GPIO_Mode_OUT,
		.GPIO_OType = GPIO_OType_PP,
		.GPIO_PuPd = GPIO_PuPd_UP
	};
	init.GPIO_Pin = pin_source;

	// Set MOSI to digital out for bit banging
	GPIO_PinAFConfig(gpio, pin_source, 0);
	GPIO_Init(gpio, &init);

	uint32_t raw_time = PIOS_DELAY_GetRaw();
	int16_t cycles = (len * 500000 / d);
	for (int16_t i = 0; i < cycles; i++) {
		GPIO_SetBits(gpio, pin_source);
		PIOS_DELAY_WaituS(d);
		GPIO_ResetBits(gpio, pin_source);
		PIOS_DELAY_WaituS(d);

		// Make sure to give other tasks time to do things
		if (PIOS_DELAY_DiffuS(raw_time) > 10000) {
			PIOS_Thread_Sleep(1);
			raw_time = PIOS_DELAY_GetRaw();
		}
	}

	GPIO_Init(gpio, (GPIO_InitTypeDef*) &openlrs_dev->cfg.spi_cfg->mosi.init);
	GPIO_PinAFConfig(gpio, pin_source, remap);

	rfm22_releaseBus(openlrs_dev);

#if defined(PIOS_LED_LINK)
	PIOS_LED_Off(PIOS_LED_LINK);
#endif /* PIOS_LED_LINK */

#if defined(PIOS_INCLUDE_WDG) && defined(PIOS_WDG_RFM22B)
		// Update the watchdog timer
		PIOS_WDG_UpdateFlag(PIOS_WDG_RFM22B);
#endif /* PIOS_WDG_RFM22B */

}


static uint8_t beaconGetRSSI(struct pios_openlrs_dev *openlrs_dev)
{
	uint16_t rssiSUM=0;

	rfmSetCarrierFrequency(openlrs_dev, openlrs_dev->beacon_frequency);
	rfm22_write_claim(openlrs_dev, RFM22_frequency_hopping_channel_select, 0); // ch 0 to avoid offset
	delay(1);
	rssiSUM+=rfmGetRSSI(openlrs_dev);
	delay(1);
	rssiSUM+=rfmGetRSSI(openlrs_dev);
	delay(1);
	rssiSUM+=rfmGetRSSI(openlrs_dev);
	delay(1);
	rssiSUM+=rfmGetRSSI(openlrs_dev);

	return rssiSUM>>2;
}

static void beacon_send(struct pios_openlrs_dev *openlrs_dev, bool static_tone)
{
	DEBUG_PRINTF(2,"beacon_send\r\n");
	rfm22_claimBus(openlrs_dev);
	openlrs_dev->it_status1 = rfm22_read(openlrs_dev, 0x03);   // read status, clear interrupt
	openlrs_dev->it_status2 = rfm22_read(openlrs_dev, 0x04);
	rfm22_write(openlrs_dev, 0x06, 0x00);    // no wakeup up, lbd,
	rfm22_write(openlrs_dev, 0x07, RF22B_PWRSTATE_READY);      // disable lbd, wakeup timer, use internal 32768,xton = 1; in ready mode
	rfm22_write(openlrs_dev, 0x09, 0x7f);  // (default) c = 12.5p
	rfm22_write(openlrs_dev, 0x0a, 0x05);
	rfm22_write(openlrs_dev, 0x0b, 0x12);    // gpio0 TX State
	rfm22_write(openlrs_dev, 0x0c, 0x15);    // gpio1 RX State
	rfm22_write(openlrs_dev, 0x0d, 0xfd);    // gpio 2 micro-controller clk output
	rfm22_write(openlrs_dev, 0x0e, 0x00);    // gpio    0, 1,2 NO OTHER FUNCTION.

	rfm22_write(openlrs_dev, 0x70, 0x2C);    // disable manchest

	rfm22_write(openlrs_dev, 0x30, 0x00);    //disable packet handling

	rfm22_write(openlrs_dev, 0x79, 0);	// start channel

	rfm22_write(openlrs_dev, 0x7a, 0x05);   // 50khz step size (10khz x value) // no hopping

	rfm22_write(openlrs_dev, 0x71, 0x12);   // trclk=[00] no clock, dtmod=[01] direct using SPI, fd8=0 eninv=0 modtyp=[10] FSK
	rfm22_write(openlrs_dev, 0x72, 0x02);   // fd (frequency deviation) 2*625Hz == 1.25kHz

	rfm22_write(openlrs_dev, 0x73, 0x00);
	rfm22_write(openlrs_dev, 0x74, 0x00);    // no offset
	rfm22_releaseBus(openlrs_dev);

	rfmSetCarrierFrequency(openlrs_dev, openlrs_dev->beacon_frequency);

	rfm22_write_claim(openlrs_dev, 0x6d, 0x07);   // 7 set max power 100mW

	delay(10);
	rfm22_write_claim(openlrs_dev, 0x07, RF22B_PWRSTATE_TX);	// to tx mode
	delay(10);

	if (static_tone) {
		uint8_t i=0;
		while (i++<20) {
			beacon_tone(openlrs_dev, 440, 1);
		}
	} else {
		//close encounters tune
		//  G, A, F, F(lower octave), C
		//octave 3:  392  440  349  175   261

		beacon_tone(openlrs_dev, 392, 1);

		rfm22_write(openlrs_dev, 0x6d, 0x05);	// 5 set mid power 25mW
		delay(10);
		beacon_tone(openlrs_dev, 440,1);

		rfm22_write(openlrs_dev, 0x6d, 0x04);	// 4 set mid power 13mW
		delay(10);
		beacon_tone(openlrs_dev, 349, 1);

		rfm22_write(openlrs_dev, 0x6d, 0x02);	// 2 set min power 3mW
		delay(10);
		beacon_tone(openlrs_dev, 175,1);

		rfm22_write(openlrs_dev, 0x6d, 0x00);	// 0 set min power 1.3mW
		delay(10);
		beacon_tone(openlrs_dev, 261, 2);
	}
	rfm22_write_claim(openlrs_dev, 0x07, RF22B_PWRSTATE_READY);
}


/*****************************************************************************
* High level OpenLRS functions
*****************************************************************************/

// TODO: these should move into device structure, or deleted
// if not useful to be reported via GCS

#define ntohl(v) (				\
	(((v) & 0xFF000000) >> 24) |		\
	(((v) & 0x00FF0000) >>  8) |		\
	(((v) & 0x0000FF00) <<  8) |		\
	(((v) & 0x000000FF) << 24))

static uint8_t pios_openlrs_bind_receive(struct pios_openlrs_dev *openlrs_dev, uint32_t timeout)
{
	uint32_t start = millis();
	uint8_t  rxb;
	init_rfm(openlrs_dev, true);
	// TODO: move openlrs_dev->rf_mode into dev structure
	openlrs_dev->rf_mode = Receive;
	to_rx_mode(openlrs_dev);
	DEBUG_PRINTF(2,"Waiting bind\r\n");

	uint32_t i = 0;

	while ((!timeout) || ((millis() - start) < timeout)) {
		PIOS_Thread_Sleep(1);
#if defined(PIOS_INCLUDE_WDG) && defined(PIOS_WDG_RFM22B)
		// Update the watchdog timer
		PIOS_WDG_UpdateFlag(PIOS_WDG_RFM22B);
#endif /* PIOS_WDG_RFM22B */

		if (i++ % 100 == 0) {
			DEBUG_PRINTF(2,"Waiting bind\r\n");

#if defined(PIOS_LED_LINK)
			PIOS_LED_Toggle(PIOS_LED_LINK);
#endif /* PIOS_LED_LINK */
		}
		if (openlrs_dev->rf_mode == Received) {

			DEBUG_PRINTF(2,"Got pkt\r\n");

			// TODO: parse data packet (write command for that)
			rfm22_claimBus(openlrs_dev);
			rfm22_assertCs(openlrs_dev);
			PIOS_SPI_TransferByte(openlrs_dev->spi_id, 0x7f);
			rxb = PIOS_SPI_TransferByte(openlrs_dev->spi_id, 0x00);
			if (rxb == 'b') {
				PIOS_SPI_TransferBlock(openlrs_dev->spi_id, OUT_FF,
			      (uint8_t*) &openlrs_dev->bind_data, sizeof(struct bind_data), NULL);
				rfm22_deassertCs(openlrs_dev);
				rfm22_releaseBus(openlrs_dev);
				
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
				if(2 <= DEBUG_LEVEL && pios_com_debug_id > 0) {
					DEBUG_PRINTF(2, "Binding settings:\r\n");
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  version: %d\r\n", openlrs_dev->bind_data.version);
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  serial_baudrate: %d\r\n", openlrs_dev->bind_data.serial_baudrate);
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  rf_frequency: %d\r\n", openlrs_dev->bind_data.rf_frequency);
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  rf_power: %d\r\n", openlrs_dev->bind_data.rf_power);
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  rf_channel_spacing: %d\r\n", openlrs_dev->bind_data.rf_channel_spacing);
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  modem_params: %d\r\n", openlrs_dev->bind_data.modem_params);
					PIOS_Thread_Sleep(10);
					DEBUG_PRINTF(2, "  flags: %d\r\n", openlrs_dev->bind_data.flags);
					PIOS_Thread_Sleep(10);

					for (uint32_t i = 0; i < MAXHOPS; i++) {
						DEBUG_PRINTF(2, "    hop channel: %d\r\n", openlrs_dev->bind_data.hopchannel[i]);
						PIOS_Thread_Sleep(10);
					}
				}
#endif /* PIOS_INCLUDE_DEBUG_CONSOLE */

				if (openlrs_dev->bind_data.version == BINDING_VERSION) {
					DEBUG_PRINTF(2,"data good\r\n");
					rxb = 'B';
					tx_packet(openlrs_dev, &rxb, 1); // ACK that we got bound

					OpenLRSData binding;
					binding.version = openlrs_dev->bind_data.version;
					binding.serial_baudrate = openlrs_dev->bind_data.serial_baudrate;
					binding.rf_frequency = openlrs_dev->bind_data.rf_frequency;
					binding.rf_magic = openlrs_dev->bind_data.rf_magic;
					binding.rf_power = openlrs_dev->bind_data.rf_power;
					binding.rf_channel_spacing = openlrs_dev->bind_data.rf_channel_spacing;
					binding.modem_params = openlrs_dev->bind_data.modem_params;
					binding.flags = openlrs_dev->bind_data.flags;
					for (uint32_t i = 0; i < OPENLRS_HOPCHANNEL_NUMELEM; i++)
						binding.hopchannel[i] = openlrs_dev->bind_data.hopchannel[i];
					binding.beacon_frequency = openlrs_dev->beacon_frequency;
					binding.beacon_delay = openlrs_dev->beacon_delay;
					binding.beacon_period = openlrs_dev->beacon_period;
					OpenLRSSet(&binding);
					UAVObjSave(OpenLRSHandle(), 0);

#if defined(PIOS_LED_LINK)
					PIOS_LED_Toggle(PIOS_LED_LINK);
#endif /* PIOS_LED_LINK */

					return 1;
				}
			} else {
				rfm22_deassertCs(openlrs_dev);
				rfm22_releaseBus(openlrs_dev);
			}
			
			openlrs_dev->rf_mode = Receive;
			rx_reset(openlrs_dev);
		}
	}
	return 0;
}

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
static void printVersion(uint16_t v)
{
	char ver[8];
	ver[0] = '0' + ((v >> 8) & 0x0f);
	ver[1] = '.';
	ver[2] = '0' + ((v >> 4) & 0x0f);
  	if (v & 0x0f) {
    	ver[3] = '.';
    	ver[4] = '0' + (v & 0x0f);
    	ver[5] = '\r';
    	ver[6] = '\n';
    	ver[7] = '\0';
    } else {
    	ver[3] = '\r';
    	ver[4] = '\n';
    	ver[5] = '\0';
    }
    DEBUG_PRINTF(2, ver);
}
#endif

static void pios_openlrs_setup(struct pios_openlrs_dev *openlrs_dev, bool bind)
{
	DEBUG_PRINTF(2,"OpenLRSng RX setup starting. Binding: %e\r\n", bind);
	PIOS_Thread_Sleep(5);
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
	printVersion(OPENLRSNG_VERSION);
#endif

	if ( bind ) {
		if (pios_openlrs_bind_receive(openlrs_dev, 0)) {
			// TODO: save binding settings bindWriteEeprom();
			DEBUG_PRINTF(2,"Saved bind data to EEPROM (not really yet -- TODO)\r\n");
		}
	}

	DEBUG_PRINTF(2,"Entering normal mode\r\n");

	init_rfm(openlrs_dev, 0);   // Configure the RFM22B's registers for normal operation
	openlrs_dev->rf_channel = 0;
	rfmSetChannel(openlrs_dev, openlrs_dev->rf_channel);

	// Count hopchannels as we need it later
	openlrs_dev->hopcount = 0;
	while ((openlrs_dev->hopcount < MAXHOPS) && (openlrs_dev->bind_data.hopchannel[openlrs_dev->hopcount] != 0)) {
		openlrs_dev->hopcount++;
	}

	//################### RX SYNC AT STARTUP #################
	openlrs_dev->rf_mode = Receive;
	to_rx_mode(openlrs_dev);

	openlrs_dev->link_acquired = 0;
	openlrs_dev->lastPacketTimeUs = micros();

	DEBUG_PRINTF(2,"OpenLRSng RX setup complete\r\n");
}

static void pios_openlrs_rx_loop(struct pios_openlrs_dev *openlrs_dev)
{
	uint32_t timeUs, timeMs;

#if defined(PIOS_INCLUDE_WDG) && defined(PIOS_WDG_RFM22B)
		// Update the watchdog timer
	PIOS_WDG_UpdateFlag(PIOS_WDG_RFM22B);
#endif /* PIOS_WDG_RFM22B */

	if (rfm22_read_claim(openlrs_dev, 0x0C) == 0) {     // detect the locked module and reboot
		DEBUG_PRINTF(2,"RX hang\r\n");
		init_rfm(openlrs_dev, 0);
		to_rx_mode(openlrs_dev);
	}

	timeUs = micros();
	timeMs = millis();

	uint8_t *tx_buf = openlrs_dev->tx_buf;  // convenient variable

	if (openlrs_dev->rf_mode == Received) {

		// Read the packet from RFM22b
		rfm22_claimBus(openlrs_dev);
		rfm22_assertCs(openlrs_dev);
		PIOS_SPI_TransferByte(openlrs_dev->spi_id, 0x7F);
		uint32_t packet_size = getPacketSize(&openlrs_dev->bind_data);
		PIOS_SPI_TransferBlock(openlrs_dev->spi_id, OUT_FF,
			      openlrs_dev->rx_buf, packet_size, NULL);
		rfm22_deassertCs(openlrs_dev);
		rfm22_releaseBus(openlrs_dev);

		openlrs_dev->lastAFCCvalue = rfmGetAFCC(openlrs_dev);

#if defined(PIOS_LED_LINK)
		PIOS_LED_Toggle(PIOS_LED_LINK);
#endif /* PIOS_LED_LINK */

		openlrs_dev->lastPacketTimeUs = timeUs;
		openlrs_dev->numberOfLostPackets = 0;
		openlrs_dev->linkQuality <<= 1;
		openlrs_dev->linkQuality |= 1;

		// TODO: updateLBeep(false);

		if ((openlrs_dev->rx_buf[0] & 0x3e) == 0x00) {
			unpackChannels(openlrs_dev->bind_data.flags & 7, openlrs_dev->ppm, openlrs_dev->rx_buf + 1);
			rescaleChannels(openlrs_dev->ppm);

			// Call the PPM received callback if it's available.
			if (openlrs_dev->openlrs_rcvr_id) {
#if defined(PIOS_INCLUDE_OPENLRS_RCVR)
				PIOS_OpenLRS_Rcvr_UpdateChannels(openlrs_dev->openlrs_rcvr_id, openlrs_dev->ppm);
#endif
			}

			//set_PPM_rssi();

			/* I think this is failsafe related
			if (openlrs_dev->rx_buf[0] & 0x01) {
				if (!fs_saved) {
					for (int16_t i = 0; i < PPM_CHANNELS; i++) {
						if (!(failsafePPM[i] & 0x1000)) {
							failsafePPM[i] = servoBits2Us(PPM[i]);
						}
					}
					failsafeSave();
					fs_saved = 1;
				}
			} else if (fs_saved) {
				fs_saved = 0;
			}
			*/
		} 
		else {
			// something else than servo data...
			if ((openlrs_dev->rx_buf[0] & 0x38) == 0x38) {
				if ((openlrs_dev->rx_buf[0] ^ tx_buf[0]) & 0x80) {
					// We got new data... (not retransmission)
					tx_buf[0] ^= 0x80; // signal that we got it
					bool rx_need_yield;
					uint8_t data_len = openlrs_dev->rx_buf[0] & 7;
					if (openlrs_dev->rx_in_cb && (data_len > 0)) {
						(openlrs_dev->rx_in_cb) (openlrs_dev->rx_in_context, &openlrs_dev->rx_buf[1], data_len, NULL, &rx_need_yield);
					}
				}
			}
		}

		// Flag to indicate ever got a link
		openlrs_dev->link_acquired |= true;
		openlrs_dev->failsafeActive = 0;

		if (openlrs_dev->bind_data.flags & TELEMETRY_MASK) {
			if ((tx_buf[0] ^ openlrs_dev->rx_buf[0]) & 0x40) {
				// resend last message
			} else {
				tx_buf[0] &= 0xc0;
				tx_buf[0] ^= 0x40; // swap sequence as we have new data

				// Check for data on serial link
				uint8_t bytes = 0;
				// Append data from the com interface if applicable.
				if (openlrs_dev->tx_out_cb) {
					// Try to get some data to send
					bool need_yield = false;
					bytes = (openlrs_dev->tx_out_cb) (openlrs_dev->tx_out_context, &tx_buf[1], 8, NULL, &need_yield);
				}

				if (bytes > 0) {
					tx_buf[0] |= (0x37 + bytes);
				} else {
					// tx_buf[0] lowest 6 bits left at 0
					tx_buf[1] = openlrs_dev->lastRSSIvalue;
					if (FlightBatteryStateHandle()) {
						FlightBatteryStateData bat;
						FlightBatteryStateGet(&bat);
						// FrSky protocol normally uses 3.3V at 255 but
						// divider from display can be set internally
						tx_buf[2] = (uint8_t) bat.Voltage / 25.0f * 255;
						tx_buf[3] = (uint8_t) bat.Current / 60.0f * 255;
					} else {
						tx_buf[2] = 0; // these bytes carry analog info. package
						tx_buf[3] = 0; // battery here
					}
					tx_buf[4] = (openlrs_dev->lastAFCCvalue >> 8);
					tx_buf[5] = openlrs_dev->lastAFCCvalue & 0xff;
					tx_buf[6] = countSetBits(openlrs_dev->linkQuality & 0x7fff);
				}
			}

			// This will block until sent
			tx_packet(openlrs_dev, tx_buf, 9);
		}

		// updateSwitches();

		openlrs_dev->rf_mode = Receive;
		rx_reset(openlrs_dev);

		openlrs_dev->willhop = 1;
	}

	// sample RSSI when packet is in the 'air'
	if ((openlrs_dev->numberOfLostPackets < 2) && (openlrs_dev->lastRSSITimeUs != openlrs_dev->lastPacketTimeUs) &&
			(timeUs - openlrs_dev->lastPacketTimeUs) > (getInterval(&openlrs_dev->bind_data) - 1500)) {

		//DEBUG_PRINTF(3,"pios_openlrs_rx_loop -- measure RSSI\r\n");

		openlrs_dev->lastRSSITimeUs = openlrs_dev->lastPacketTimeUs;
		openlrs_dev->lastRSSIvalue = rfmGetRSSI(openlrs_dev); // Read the RSSI value

		DEBUG_PRINTF(2,"RSSI: %d\r\n", openlrs_dev->lastRSSIvalue);
	}

	if (openlrs_dev->link_acquired) {
		if ((openlrs_dev->numberOfLostPackets < openlrs_dev->hopcount) && ((timeUs - openlrs_dev->lastPacketTimeUs) > (getInterval(&openlrs_dev->bind_data) + 1000))) {
			DEBUG_PRINTF(2,"Lost packet: %d\r\n", openlrs_dev->numberOfLostPackets);
			// we lost packet, hop to next channel
			openlrs_dev->linkQuality <<= 1;
			openlrs_dev->willhop = 1;
			if (openlrs_dev->numberOfLostPackets == 0) {
				openlrs_dev->linkLossTimeMs = timeMs;
				openlrs_dev->nextBeaconTimeMs = 0;
			}
			openlrs_dev->numberOfLostPackets++;
			openlrs_dev->lastPacketTimeUs += getInterval(&openlrs_dev->bind_data);
			openlrs_dev->willhop = 1;
			//updateLBeep(true);
			// TODO: set_RSSI_output();
		} else if ((openlrs_dev->numberOfLostPackets >= openlrs_dev->hopcount) && ((timeUs - openlrs_dev->lastPacketTimeUs) > (getInterval(&openlrs_dev->bind_data) * openlrs_dev->hopcount))) {
			// hop slowly to allow resync with TX
			openlrs_dev->linkQuality = 0;
			openlrs_dev->willhop = 1;
			// TODO: set_RSSI_output();
			openlrs_dev->lastPacketTimeUs = timeUs;
		}

		if (openlrs_dev->numberOfLostPackets) {

#if defined(PIOS_LED_LINK)
			PIOS_LED_Off(PIOS_LED_LINK);
#endif /* PIOS_LED_LINK */

			if (openlrs_dev->failsafeDelay &&
				!openlrs_dev->failsafeActive && 
				((timeMs - openlrs_dev->linkLossTimeMs) > ((uint32_t) openlrs_dev->failsafeDelay * 1000)))
			{
				DEBUG_PRINTF(2,"Failsafe activated: %d %d\r\n", timeMs, openlrs_dev->linkLossTimeMs);
				openlrs_dev->failsafeActive = 1;
				//failsafeApply();
				openlrs_dev->nextBeaconTimeMs = (timeMs + 1000UL * openlrs_dev->beacon_period) | 1; //beacon activating...
			}

			if ((openlrs_dev->beacon_frequency) && (openlrs_dev->nextBeaconTimeMs) &&
					((timeMs - openlrs_dev->nextBeaconTimeMs) < 0x80000000)) {

				DEBUG_PRINTF(2,"Beacon time: %d\r\n", openlrs_dev->nextBeaconTimeMs);
				// Only beacon when disarmed
				uint8_t armed;
				FlightStatusArmedGet(&armed);
				if (armed == FLIGHTSTATUS_ARMED_DISARMED) {
					beacon_send(openlrs_dev, false); // play cool tune
					init_rfm(openlrs_dev, 0);   // go back to normal RX
					rx_reset(openlrs_dev);
					openlrs_dev->nextBeaconTimeMs = (timeMs +  1000UL * openlrs_dev->beacon_period) | 1; // avoid 0 in time
				}
			}
		}

	} else {
		// Waiting for first packet, hop slowly
		if ((timeUs - openlrs_dev->lastPacketTimeUs) > (getInterval(&openlrs_dev->bind_data) * openlrs_dev->hopcount)) {
			//DEBUG_PRINTF(3,"pios_openlrs_rx_loop - slow hop\r\n");
			openlrs_dev->lastPacketTimeUs = timeUs;
			openlrs_dev->willhop = 1;
		}
	}

	if (openlrs_dev->willhop == 1) {
		openlrs_dev->rf_channel++;

		if ((openlrs_dev->rf_channel == MAXHOPS) || (openlrs_dev->bind_data.hopchannel[openlrs_dev->rf_channel] == 0)) {
			openlrs_dev->rf_channel = 0;
		}

		if ((openlrs_dev->beacon_frequency) && (openlrs_dev->nextBeaconTimeMs)) {
			// Listen for RSSI on beacon channel briefly for 'trigger'
			uint8_t brssi = beaconGetRSSI(openlrs_dev);
			if (brssi > ((openlrs_dev->beacon_rssi_avg>>2) + 20)) {
				openlrs_dev->nextBeaconTimeMs = timeMs + 1000L;
			}
			openlrs_dev->beacon_rssi_avg = (openlrs_dev->beacon_rssi_avg * 3 + brssi * 4) >> 2;

			rfmSetCarrierFrequency(openlrs_dev, openlrs_dev->bind_data.rf_frequency);
		}

		rfmSetChannel(openlrs_dev, openlrs_dev->rf_channel);
		rx_reset(openlrs_dev);
		openlrs_dev->willhop = 0;
	}
}

/*****************************************************************************
* PPM Code
*****************************************************************************/

/**
 * Register a OpenLRS_Rcvr interface to inform of PPM packets
 *
 * @param[in] rfm22b_dev     The RFM22B device ID.
 * @param[in] rfm22b_rcvr_id The receiver device to inform of PPM packets
 */
void PIOS_OpenLRS_RegisterRcvr(uintptr_t openlrs_id, uintptr_t openlrs_rcvr_id)
{
	struct pios_openlrs_dev *openlrs_dev =
	    (struct pios_openlrs_dev *)openlrs_id;

	if (!pios_openlrs_validate(openlrs_dev)) {
		return;
	}

	openlrs_dev->openlrs_rcvr_id = openlrs_rcvr_id;
}

/*****************************************************************************
* Task and device setup
*****************************************************************************/

static void pios_openlrs_task(void *parameters);

//! Global device handle, required for IRQ handler
static struct pios_openlrs_dev * g_openlrs_dev;

/**
 * Initialise an RFM22B device
 *
 * @param[out] rfm22b_id  A pointer to store the device ID in.
 * @param[in] spi_id  The SPI bus index.
 * @param[in] slave_num  The SPI bus slave number.
 * @param[in] cfg  The device configuration.
 */
int32_t PIOS_OpenLRS_Init(uintptr_t * openlrs_id, uint32_t spi_id,
			 uint32_t slave_num,
			 const struct pios_openlrs_cfg *cfg)
{
	PIOS_DEBUG_Assert(rfm22b_id);
	PIOS_DEBUG_Assert(cfg);

	// Allocate the device structure.
	struct pios_openlrs_dev *openlrs_dev = pios_openlrs_alloc();
	if (!openlrs_dev) {
		return -1;
	}
	*openlrs_id = (uint32_t) openlrs_dev;
	g_openlrs_dev = openlrs_dev;

	// Store the SPI handle
	openlrs_dev->slave_num = slave_num;
	openlrs_dev->spi_id = spi_id;

	// Before initializing everything, make sure device found
	uint8_t device_type = rfm22_read(openlrs_dev, RFM22_DEVICE_TYPE) & RFM22_DT_MASK;
	if (device_type != 0x08)
		return -1;

	// Initialize the com callbacks.
	openlrs_dev->rx_in_cb = NULL;
	openlrs_dev->tx_out_cb = NULL;

	// Initialzie the PPM callback.
	openlrs_dev->openlrs_rcvr_id = 0;

	OpenLRSInitialize();
	OpenLRSData binding;
	OpenLRSGet(&binding);
	if (binding.version == BINDING_VERSION) {
		openlrs_dev->bind_data.version = binding.version;
		openlrs_dev->bind_data.serial_baudrate = binding.serial_baudrate;
		openlrs_dev->bind_data.rf_frequency = binding.rf_frequency;
		openlrs_dev->bind_data.rf_magic = binding.rf_magic;
		openlrs_dev->bind_data.rf_power = binding.rf_power;
		openlrs_dev->bind_data.rf_channel_spacing = binding.rf_channel_spacing;
		openlrs_dev->bind_data.modem_params = binding.modem_params;
		openlrs_dev->bind_data.flags = binding.flags;
		for (uint32_t i = 0; i < OPENLRS_HOPCHANNEL_NUMELEM; i++)
			openlrs_dev->bind_data.hopchannel[i] = binding.hopchannel[i];
	}

	// Copy beacon settings over
	openlrs_dev->beacon_frequency = binding.beacon_frequency;
	openlrs_dev->beacon_delay = binding.beacon_delay;
	openlrs_dev->beacon_period = binding.beacon_period;

	// Hardcode failsafe delay
	openlrs_dev->failsafeDelay = 200;

	// Bind the configuration to the device instance
	openlrs_dev->cfg = *cfg;

	// Initialize the external interrupt.
	PIOS_EXTI_Init(cfg->exti_cfg);

	// Register the watchdog timer for the radio driver task
#if defined(PIOS_INCLUDE_WDG) && defined(PIOS_WDG_RFM22B)
	PIOS_WDG_RegisterFlag(PIOS_WDG_RFM22B);
#endif /* PIOS_WDG_RFM22B */

	// Start the driver task.  This task controls the radio state machine and removed all of the IO from the IRQ handler.
	openlrs_dev->taskHandle = PIOS_Thread_Create(pios_openlrs_task, "PIOS_OpenLRS_Task", STACK_SIZE_BYTES, (void *)openlrs_dev, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_MODEMRX, openlrs_dev->taskHandle);

	return 0;
}

/**
 * The task that controls the radio state machine.
 *
 * @param[in] paramters  The task parameters.
 */
static void pios_openlrs_task(void *parameters)
{
	struct pios_openlrs_dev *openlrs_dev = (struct pios_openlrs_dev *)parameters;

	if (!pios_openlrs_validate(openlrs_dev)) {
		return;
	}

	if (openlrs_dev->bind_data.version == BINDING_VERSION)
		pios_openlrs_setup(openlrs_dev, false);
	else
		pios_openlrs_setup(openlrs_dev, true);

	DEBUG_PRINTF(2, "Setup complete\r\n");

	while(1) {
#if defined(PIOS_INCLUDE_WDG) && defined(PIOS_WDG_RFM22B)
		// Update the watchdog timer
		PIOS_WDG_UpdateFlag(PIOS_WDG_RFM22B);
#endif /* PIOS_WDG_RFM22B */

		uint32_t time_till_measure_rssi_us  = (getInterval(&openlrs_dev->bind_data) - 1500) + openlrs_dev->lastPacketTimeUs - micros();
		uint32_t delay = time_till_measure_rssi_us / 1000;
		uint32_t max_delay = getInterval(&openlrs_dev->bind_data) / 2000;
		if (delay > max_delay) delay = max_delay;
		PIOS_Semaphore_Take(openlrs_dev->sema_isr, delay);

		// Process incoming radio data.
		pios_openlrs_rx_loop(openlrs_dev);
	}
}

bool PIOS_OpenLRS_EXT_Int(void)
{
	struct pios_openlrs_dev *openlrs_dev = g_openlrs_dev;
	if (!pios_openlrs_validate(openlrs_dev))
		return false;

	if (openlrs_dev->rf_mode == Transmit) {
		openlrs_dev->rf_mode = Transmitted;
	}
	else if (openlrs_dev->rf_mode == Receive) {
		openlrs_dev->rf_mode = Received;
	}

	// Indicate to main task that an ISR occurred
	bool woken = false;
	PIOS_Semaphore_Give_FromISR(openlrs_dev->sema_isr, &woken);

	return woken;
}


/**
 * Allocate the device structure
 */
static struct pios_openlrs_dev *pios_openlrs_alloc(void)
{
	struct pios_openlrs_dev *openlrs_dev;

	openlrs_dev = (struct pios_openlrs_dev *)PIOS_malloc(sizeof(*openlrs_dev));
	openlrs_dev->spi_id = 0;
	if (!openlrs_dev) {
		return NULL;
	}

	// Create the ISR signal
	openlrs_dev->sema_isr = PIOS_Semaphore_Create();
	if (!openlrs_dev->sema_isr) {
		PIOS_free(openlrs_dev);
		return NULL;
	}

	openlrs_dev->magic = PIOS_OPENLRS_DEV_MAGIC;
	return openlrs_dev;
}


/**
 * Validate that the device structure is valid.
 *
 * @param[in] openlrs_dev  The OpenLRS device structure pointer.
 */
static bool pios_openlrs_validate(struct pios_openlrs_dev *openlrs_dev)
{
	return openlrs_dev != NULL
	    && openlrs_dev->magic == PIOS_OPENLRS_DEV_MAGIC;
}

/*****************************************************************************
* SPI Read/Write Functions
*****************************************************************************/

/**
 * Assert the chip select line.
 *
 * @param[in] rfm22b_dev  The RFM22B device.
 */
static void rfm22_assertCs(struct pios_openlrs_dev *openlrs_dev)
{
	PIOS_DELAY_WaituS(1);
	if (openlrs_dev->spi_id != 0) {
		PIOS_SPI_RC_PinSet(openlrs_dev->spi_id,
				   openlrs_dev->slave_num, 0);
	}
}

/**
 * Deassert the chip select line.
 *
 * @param[in] rfm22b_dev  The RFM22B device structure pointer.
 */
static void rfm22_deassertCs(struct pios_openlrs_dev *openlrs_dev)
{
	if (openlrs_dev->spi_id != 0) {
		PIOS_SPI_RC_PinSet(openlrs_dev->spi_id,
				   openlrs_dev->slave_num, 1);
	}
}

/**
 * Claim the SPI bus.
 *
 * @param[in] rfm22b_dev  The RFM22B device structure pointer.
 */
static void rfm22_claimBus(struct pios_openlrs_dev *openlrs_dev)
{
	if (openlrs_dev->spi_id != 0) {
		PIOS_SPI_ClaimBus(openlrs_dev->spi_id);
	}
}

/**
 * Release the SPI bus.
 *
 * @param[in] rfm22b_dev  The RFM22B device structure pointer.
 */
static void rfm22_releaseBus(struct pios_openlrs_dev *openlrs_dev)
{
	if (openlrs_dev->spi_id != 0) {
		PIOS_SPI_ReleaseBus(openlrs_dev->spi_id);
	}
}

/**
 * Claim the semaphore and write a byte to a register
 *
 * @param[in] rfm22b_dev  The RFM22B device.
 * @param[in] addr The address to write to
 * @param[in] data The datat to write to that address
 */
static void rfm22_write_claim(struct pios_openlrs_dev *openlrs_dev,
			      uint8_t addr, uint8_t data)
{
	rfm22_claimBus(openlrs_dev);
	rfm22_assertCs(openlrs_dev);
	uint8_t buf[2] = { addr | 0x80, data };
	PIOS_SPI_TransferBlock(openlrs_dev->spi_id, buf, NULL, sizeof(buf),
			       NULL);
	rfm22_deassertCs(openlrs_dev);
	rfm22_releaseBus(openlrs_dev);
}

/**
 * Claim the semaphore and write a byte to a register
 *
 * @param[in] rfm22b_dev  The RFM22B device.
 * @param[in] addr The address to write to
 * @param[in] data The datat to write to that address
 */
static uint8_t rfm22_read_claim(struct pios_openlrs_dev *openlrs_dev,
			      uint8_t addr)
{
	uint8_t out[2] = { addr & 0x7F, 0xFF };
	uint8_t in[2];

	rfm22_claimBus(openlrs_dev);
	rfm22_assertCs(openlrs_dev);
	PIOS_SPI_TransferBlock(openlrs_dev->spi_id, out, in, sizeof(out),
			       NULL);
	rfm22_deassertCs(openlrs_dev);
	rfm22_releaseBus(openlrs_dev);
	return in[1];
}

/**
 * Write a byte to a register without claiming the semaphore
 *
 * @param[in] rfm22b_dev  The RFM22B device.
 * @param[in] addr The address to write to
 * @param[in] data The datat to write to that address
 */
static void rfm22_write(struct pios_openlrs_dev *openlrs_dev, uint8_t addr,
			uint8_t data)
{
	rfm22_assertCs(openlrs_dev);
	uint8_t buf[2] = { addr | 0x80, data };
	PIOS_SPI_TransferBlock(openlrs_dev->spi_id, buf, NULL, sizeof(buf),
			       NULL);
	rfm22_deassertCs(openlrs_dev);
}

/**
 * Read a byte from an RFM22b register without claiming the bus
 *
 * @param[in] rfm22b_dev  The RFM22B device structure pointer.
 * @param[in] addr The address to read from
 * @return Returns the result of the register read
 */
static uint8_t rfm22_read(struct pios_openlrs_dev *openlrs_dev, uint8_t addr)
{
	uint8_t out[2] = { addr & 0x7F, 0xFF };
	uint8_t in[2];

	rfm22_assertCs(openlrs_dev);
	PIOS_SPI_TransferBlock(openlrs_dev->spi_id, out, in, sizeof(out),
			       NULL);
	rfm22_deassertCs(openlrs_dev);
	return in[1];
}

#endif /* PIOS_INCLUDE_OPENLRS */

/**
 * @}
 * @}
 */
