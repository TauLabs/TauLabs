#ifndef PIOS_PORT_H
#define PIOS_PORT_H

#include <uavobjectmanager.h> /* XXX TODO */
#include <hwshared.h>
#include <pios_usart_priv.h>
#include <pios_sbus_priv.h>
#include <pios_dsm_priv.h>
#include <pios_ppm_priv.h>
#include <pios_i2c_priv.h>
#include <pios_usb_cdc_priv.h>
#include <pios_usb_hid_priv.h>
#include <pios_openlrs_priv.h>
#include <pios_rfm22b_priv.h>

void PIOS_HAL_panic(uint32_t led_id, int32_t code);
void PIOS_HAL_configure_port(HwSharedPortTypesOptions port_type,
		const struct pios_usart_cfg *usart_port_cfg,
		const struct pios_com_driver *com_driver,
		uint32_t *i2c_id,
		const struct pios_i2c_adapter_cfg *i2c_cfg,
		const struct pios_ppm_cfg *ppm_cfg,
		uint32_t led_id,
		/* TODO: future work to factor most of these away */
		const struct pios_usart_cfg *usart_dsm_hsum_cfg,
		const struct pios_dsm_cfg *dsm_cfg,
		uintptr_t dsm_channelgroup,
		int dsm_bind,
		const struct pios_usart_cfg *sbus_rcvr_cfg,
		const struct pios_sbus_cfg *sbus_cfg,
		bool sbus_toggle
		);

void PIOS_HAL_configure_CDC(HwSharedUSB_VCPPortOptions port_type,
		uintptr_t usb_id,
		const struct pios_usb_cdc_cfg *cdc_cfg);

void PIOS_HAL_configure_HID(HwSharedUSB_HIDPortOptions port_type,
		uintptr_t usb_id,
		const struct pios_usb_hid_cfg *hid_cfg);

void PIOS_HAL_configure_RFM22B(HwSharedRadioPortOptions radio_type,
                uint8_t board_type, uint8_t board_rev,
                HwSharedMaxRfPowerOptions max_power,
                HwSharedMaxRfSpeedOptions max_speed,
                const struct pios_openlrs_cfg *openlrs_cfg,
                const struct pios_rfm22b_cfg *rfm22b_cfg,
                uint8_t min_chan, uint8_t max_chan, uint32_t coord_id);

#endif
