#ifndef PIOS_HAL_H
#define PIOS_HAL_H

#include <uavobjectmanager.h> /* XXX TODO */
#include <hwshared.h>
#include <pios_usart_priv.h>
#include <pios_sbus_priv.h>
#include <pios_dsm_priv.h>
#include <pios_ppm_priv.h>
#include <pios_pwm_priv.h>
#include <pios_i2c_priv.h>
#include <pios_usb_cdc_priv.h>
#include <pios_usb_hid_priv.h>

#if defined(PIOS_INCLUDE_RFM22B)
#include <pios_rfm22b_priv.h>
#include <pios_openlrs_priv.h>
#endif /* PIOS_INCLUDE_RFM22B */

/* One slot per selectable receiver group.
 *  eg. PWM, PPM, GCS, SPEKTRUM1, SPEKTRUM2, SBUS
 * NOTE: No slot in this map for NONE.
 */
extern uintptr_t pios_rcvr_group_map[];

#if defined(PIOS_INCLUDE_SBUS) || defined(PIOS_INCLUDE_DSM) || defined(PIOS_INCLUDE_HOTT) || defined(PIOS_INCLUDE_GPS) || defined(PIOS_INCLUDE_RFM22B) || defined(PIOS_INCLUDE_USB_CDC) || defined(PIOS_INCLUDE_USB_HID) || defined(PIOS_INCLUDE_MAVLINK)

#ifndef PIOS_INCLUDE_COM
#error Options defined that require PIOS_INCLUDE_COM!
#endif

/* This one is a slight overreach; not all of the above requires this but close */
#ifndef PIOS_INCLUDE_USART
#error Options defined that require PIOS_INCLUDE_USART!
#endif

#endif

void PIOS_HAL_ConfigureUsart(const struct pios_usart_cfg *usart_port_cfg,
		size_t rx_buf_len, size_t tx_buf_len,
		const struct pios_com_driver *com_driver, uintptr_t *com_id);

void PIOS_HAL_Panic(uint32_t led_id, int32_t code);

void PIOS_HAL_ConfigurePort(HwSharedPortTypesOptions port_type,
		const struct pios_usart_cfg *usart_port_cfg,
		const struct pios_usart_cfg *usart_frsky_port_cfg,
		const struct pios_com_driver *com_driver,
		uint32_t *i2c_id,
		const struct pios_i2c_adapter_cfg *i2c_cfg,
		const struct pios_ppm_cfg *ppm_cfg,
		const struct pios_pwm_cfg *pwm_cfg,
		uint32_t led_id,
		/* TODO: future work to factor most of these away */
		const struct pios_usart_cfg *usart_dsm_hsum_cfg,
		const struct pios_dsm_cfg *dsm_cfg,
		HwSharedDSMxModeOptions dsm_mode,
		const struct pios_usart_cfg *sbus_rcvr_cfg,
		const struct pios_sbus_cfg *sbus_cfg,
		bool sbus_toggle);

void PIOS_HAL_ConfigureCDC(HwSharedUSB_VCPPortOptions port_type,
		uintptr_t usb_id,
		const struct pios_usb_cdc_cfg *cdc_cfg);

void PIOS_HAL_ConfigureHID(HwSharedUSB_HIDPortOptions port_type,
		uintptr_t usb_id,
		const struct pios_usb_hid_cfg *hid_cfg);

#if defined(PIOS_INCLUDE_RFM22B)
void PIOS_HAL_ConfigureRFM22B(HwSharedRadioPortOptions radio_type,
		uint8_t board_type, uint8_t board_rev,
		HwSharedMaxRfPowerOptions max_power,
		HwSharedMaxRfSpeedOptions max_speed,
		HwSharedRfBandOptions rf_band,
		const struct pios_openlrs_cfg *openlrs_cfg,
		const struct pios_rfm22b_cfg *rfm22b_cfg,
		uint8_t min_chan, uint8_t max_chan, uint32_t coord_id,
		HwSharedPortTypesOptions port_type,
		int status_inst);
#endif /* PIOS_INCLUDE_RFM22B */

#endif
