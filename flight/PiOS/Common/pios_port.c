#include <pios.h>
#include <pios_port.h>

#include <pios_com_priv.h>
#include <pios_rcvr_priv.h>
#include <pios_openlrs_rcvr_priv.h>
#include <pios_rfm22b_rcvr_priv.h>
#include <pios_hsum_priv.h>

#include <manualcontrolsettings.h>

extern uintptr_t pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

uintptr_t pios_com_gps_id;
uintptr_t pios_com_vcp_id;
uintptr_t pios_com_bridge_id;
uintptr_t pios_com_overo_id;
uintptr_t pios_com_mavlink_id;
uintptr_t pios_com_hott_id;
uintptr_t pios_com_frsky_sensor_hub_id;
uintptr_t pios_com_frsky_sport_id;
uintptr_t pios_com_lighttelemetry_id;
uintptr_t pios_com_picoc_id;
uintptr_t pios_com_rf_id;
uintptr_t pios_com_telem_usb_id;
uintptr_t pios_com_telem_rf_id;



#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
// PENDING FACTOR VCP STUFF INTO HERE
//uintptr_t pios_com_debug_id;
#endif  /* PIOS_INCLUDE_DEBUG_CONSOLE */

#define PIOS_COM_TELEM_RF_RX_BUF_LEN 512
#define PIOS_COM_TELEM_RF_TX_BUF_LEN 512

#define PIOS_COM_GPS_RX_BUF_LEN 32
#define PIOS_COM_GPS_TX_BUF_LEN 16

#define PIOS_COM_TELEM_USB_RX_BUF_LEN 65
#define PIOS_COM_TELEM_USB_TX_BUF_LEN 65

#define PIOS_COM_BRIDGE_RX_BUF_LEN 65
#define PIOS_COM_BRIDGE_TX_BUF_LEN 12

#define PIOS_COM_MAVLINK_TX_BUF_LEN 128

#define PIOS_COM_HOTT_RX_BUF_LEN 16
#define PIOS_COM_HOTT_TX_BUF_LEN 16

#define PIOS_COM_FRSKYSENSORHUB_TX_BUF_LEN 128

#define PIOS_COM_LIGHTTELEMETRY_TX_BUF_LEN 19

#define PIOS_COM_PICOC_RX_BUF_LEN 128
#define PIOS_COM_PICOC_TX_BUF_LEN 128

#define PIOS_COM_FRSKYSPORT_TX_BUF_LEN 16
#define PIOS_COM_FRSKYSPORT_RX_BUF_LEN 16

#define PIOS_COM_RFM22B_RF_RX_BUF_LEN 512
#define PIOS_COM_RFM22B_RF_TX_BUF_LEN 512

void PIOS_HAL_panic(uint32_t led_id, int32_t code) {
        while(1){
                for (int32_t i = 0; i < code; i++) {
                        PIOS_WDG_Clear();
                        PIOS_LED_Toggle(led_id);
                        PIOS_DELAY_WaitmS(200);
                        PIOS_WDG_Clear();
                        PIOS_LED_Toggle(led_id);
                        PIOS_DELAY_WaitmS(200);
                }
                PIOS_DELAY_WaitmS(200);
                PIOS_WDG_Clear();
                PIOS_DELAY_WaitmS(200);
                PIOS_WDG_Clear();
                PIOS_DELAY_WaitmS(100);
                PIOS_WDG_Clear();
        }
}

#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
static void PIOS_HAL_configure_com (const struct pios_usart_cfg *usart_port_cfg, size_t rx_buf_len, size_t tx_buf_len,
                const struct pios_com_driver *com_driver, uintptr_t *pios_com_id)
{
        uintptr_t pios_usart_id;
        if (PIOS_USART_Init(&pios_usart_id, usart_port_cfg)) {
                PIOS_Assert(0);
        }

        uint8_t * rx_buffer;
        if (rx_buf_len > 0) {
                rx_buffer = (uint8_t *) PIOS_malloc(rx_buf_len);
                PIOS_Assert(rx_buffer);
        } else {
                rx_buffer = NULL;
        }

        uint8_t * tx_buffer;
        if (tx_buf_len > 0) {
                tx_buffer = (uint8_t *) PIOS_malloc(tx_buf_len);
                PIOS_Assert(tx_buffer);
        } else {
                tx_buffer = NULL;
        }

        if (PIOS_COM_Init(pios_com_id, com_driver, pios_usart_id,
                                rx_buffer, rx_buf_len,
                                tx_buffer, tx_buf_len)) {
                PIOS_Assert(0);
        }
}
#endif  /* PIOS_INCLUDE_USART && PIOS_INCLUDE_COM */

#ifdef PIOS_INCLUDE_DSM
static void PIOS_HAL_configure_dsm(const struct pios_usart_cfg *pios_usart_dsm_cfg,
                const struct pios_dsm_cfg *pios_dsm_cfg,
                const struct pios_com_driver *pios_usart_com_driver,
                ManualControlSettingsChannelGroupsOptions channelgroup,
		int bind)
{
        uintptr_t pios_usart_dsm_id;
        if (PIOS_USART_Init(&pios_usart_dsm_id, pios_usart_dsm_cfg)) {
                PIOS_Assert(0);
        }
        
        uintptr_t pios_dsm_id;
        if (PIOS_DSM_Init(&pios_dsm_id, pios_dsm_cfg, pios_usart_com_driver,
                        pios_usart_dsm_id, bind)) {
                PIOS_Assert(0);
        }
        
        uintptr_t pios_dsm_rcvr_id;
        if (PIOS_RCVR_Init(&pios_dsm_rcvr_id, &pios_dsm_rcvr_driver, pios_dsm_id)) {
                PIOS_Assert(0);
        }
        pios_rcvr_group_map[channelgroup] = pios_dsm_rcvr_id;
}

#endif

#ifdef PIOS_INCLUDE_HSUM
static void PIOS_HAL_configure_hsum(const struct pios_usart_cfg *pios_usart_hsum_cfg,
                const struct pios_com_driver *pios_usart_com_driver,enum pios_hsum_proto *proto,
                ManualControlSettingsChannelGroupsOptions channelgroup)
{
        uintptr_t pios_usart_hsum_id;
        if (PIOS_USART_Init(&pios_usart_hsum_id, pios_usart_hsum_cfg)) {
                PIOS_Assert(0);
        }
        
        uintptr_t pios_hsum_id;
        if (PIOS_HSUM_Init(&pios_hsum_id, pios_usart_com_driver,
                           pios_usart_hsum_id, *proto)) {
                PIOS_Assert(0);
        }
        
        uintptr_t pios_hsum_rcvr_id;
        if (PIOS_RCVR_Init(&pios_hsum_rcvr_id, &pios_hsum_rcvr_driver, pios_hsum_id)) {
                PIOS_Assert(0);
        }
        pios_rcvr_group_map[channelgroup] = pios_hsum_rcvr_id;
}
#endif

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
	)
{
	uintptr_t port_driver_id;
	uintptr_t *target=NULL, *target2=NULL;;

        switch (port_type) {
		case HWSHARED_PORTTYPES_I2C:
#if defined(PIOS_INCLUDE_I2C)
			if (i2c_id && i2c_cfg) {
				if (PIOS_I2C_Init(i2c_id, i2c_cfg)) {
					PIOS_Assert(0);
				}
				if (PIOS_I2C_CheckClear(*i2c_id) != 0)
					PIOS_HAL_panic(led_id, 6);
			}
#endif  /* PIOS_INCLUDE_I2C */
			break;
		case HWSHARED_PORTTYPES_PPM:
#if defined(PIOS_INCLUDE_PPM)
			if (ppm_cfg) { 
				uintptr_t pios_ppm_id;
				PIOS_PPM_Init(&pios_ppm_id, ppm_cfg);

				uintptr_t pios_ppm_rcvr_id;
				if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
					PIOS_Assert(0);
				}
				pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM] = pios_ppm_rcvr_id;
			}
#endif  /* PIOS_INCLUDE_PPM */
			break;

		case HWSHARED_PORTTYPES_DISABLED:
			break;
		case HWSHARED_PORTTYPES_TELEMETRY:
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_telem_rf_id;
			break;
		case HWSHARED_PORTTYPES_GPS:
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_GPS_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_gps_id;
			break;
		case HWSHARED_PORTTYPES_DSM:
#if defined(PIOS_INCLUDE_DSM)
			if (dsm_cfg && usart_dsm_hsum_cfg) {
				PIOS_HAL_configure_dsm(usart_dsm_hsum_cfg, dsm_cfg, com_driver, dsm_channelgroup, dsm_bind);
			}
#endif  /* PIOS_INCLUDE_DSM */
			break;
		case HWSHARED_PORTTYPES_SBUS:
#if defined(PIOS_INCLUDE_SBUS) && defined(PIOS_INCLUDE_USART)
			if (sbus_cfg && sbus_rcvr_cfg) {
				uintptr_t usart_sbus_id;
				if (PIOS_USART_Init(&usart_sbus_id, sbus_rcvr_cfg)) {
					PIOS_Assert(0);
				}
				uintptr_t sbus_id;
				if (PIOS_SBus_Init(&sbus_id, sbus_cfg, com_driver, usart_sbus_id)) {
					PIOS_Assert(0);
				}
				uintptr_t sbus_rcvr_id;
				if (PIOS_RCVR_Init(&sbus_rcvr_id, &pios_sbus_rcvr_driver, sbus_id)) {
					PIOS_Assert(0);
				}
				pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_SBUS] = sbus_rcvr_id;
			}
#endif  /* PIOS_INCLUDE_SBUS */
			break;
		case HWSHARED_PORTTYPES_HOTTSUMD:
		case HWSHARED_PORTTYPES_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
			if (usart_dsm_hsum_cfg) { 
				enum pios_hsum_proto proto;
				switch (port_type) {
					case HWSHARED_PORTTYPES_HOTTSUMD:
						proto = PIOS_HSUM_PROTO_SUMD;
						break;
					case HWSHARED_PORTTYPES_HOTTSUMH:
						proto = PIOS_HSUM_PROTO_SUMH;
						break;
					default:
						PIOS_Assert(0);
						break;
				}
				PIOS_HAL_configure_hsum(usart_dsm_hsum_cfg, com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM);
			}
#endif  /* PIOS_INCLUDE_HSUM */
			break;
		case HWSHARED_PORTTYPES_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			PIOS_HAL_configure_com(usart_port_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_debug_id;
#endif  /* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWSHARED_PORTTYPES_COMBRIDGE:
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_bridge_id;
			break;
		case HWSHARED_PORTTYPES_MAVLINKTX:
#if defined(PIOS_INCLUDE_MAVLINK)
			PIOS_HAL_configure_com(usart_port_cfg, 0, PIOS_COM_MAVLINK_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_mavlink_id;
#endif          /* PIOS_INCLUDE_MAVLINK */
			break;
		case HWSHARED_PORTTYPES_MAVLINKTX_GPS_RX:
#if defined(PIOS_INCLUDE_MAVLINK)
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_MAVLINK_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_mavlink_id;
			target2 = &pios_com_gps_id;
#endif          /* PIOS_INCLUDE_MAVLINK */
			break;
		case HWSHARED_PORTTYPES_HOTTTELEMETRY:
#if defined(PIOS_INCLUDE_HOTT) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_HOTT_RX_BUF_LEN, PIOS_COM_HOTT_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_hott_id;
#endif /* PIOS_INCLUDE_HOTT */
			break;
		case HWSHARED_PORTTYPES_FRSKYSENSORHUB:
#if defined(PIOS_INCLUDE_FRSKY_SENSOR_HUB) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_HAL_configure_com(usart_port_cfg, 0, PIOS_COM_FRSKYSENSORHUB_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_frsky_sensor_hub_id;
#endif /* PIOS_INCLUDE_FRSKY_SENSOR_HUB */
			break;
		case HWSHARED_PORTTYPES_FRSKYSPORTTELEMETRY:
#if defined(PIOS_INCLUDE_FRSKY_SPORT_TELEMETRY) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_FRSKYSPORT_RX_BUF_LEN, PIOS_COM_FRSKYSPORT_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_frsky_sport_id;
#endif /* PIOS_INCLUDE_FRSKY_SPORT_TELEMETRY */
			break;
		case HWSHARED_PORTTYPES_LIGHTTELEMETRYTX:
#if defined(PIOS_INCLUDE_LIGHTTELEMETRY)
			PIOS_HAL_configure_com(usart_port_cfg, 0, PIOS_COM_LIGHTTELEMETRY_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_lighttelemetry_id;
#endif 
			break;
		case HWSHARED_PORTTYPES_PICOC:
#if defined(PIOS_INCLUDE_PICOC) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_HAL_configure_com(usart_port_cfg, PIOS_COM_PICOC_RX_BUF_LEN, PIOS_COM_PICOC_TX_BUF_LEN, com_driver, &port_driver_id);
			target = &pios_com_picoc_id;
#endif /* PIOS_INCLUDE_PICOC */
			break;
        } /* port_type */

        if (port_type != HWSHARED_PORTTYPES_SBUS && sbus_toggle) {
                GPIO_Init(sbus_cfg->inv.gpio, (GPIO_InitTypeDef*)&sbus_cfg->inv.init);
                GPIO_WriteBit(sbus_cfg->inv.gpio, sbus_cfg->inv.init.GPIO_Pin, sbus_cfg->gpio_inv_disable);
        }

	// TODO: catch configuration errors of duplicated channels here
	if (target) {
		*target = port_driver_id;
		if (target2) {
			*target2 = port_driver_id;
		}
	}

}

#if defined(PIOS_INCLUDE_USB_CDC)
void PIOS_HAL_configure_CDC(HwSharedUSB_VCPPortOptions port_type,
		uintptr_t usb_id,
		const struct pios_usb_cdc_cfg *cdc_cfg) {
	uintptr_t pios_usb_cdc_id;

	// TODO: Should we actually do this if disabled???
	// TODO: Simplify-- enabling CDC doesn't make sense without INCLUDE_COM

	if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, cdc_cfg, usb_id)) {
		PIOS_Assert(0);
	}

	switch (port_type) {
		case HWSHARED_USB_VCPPORT_DISABLED:
			break;
		case HWSHARED_USB_VCPPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
			{
				uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
				uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
				PIOS_Assert(rx_buffer);
				PIOS_Assert(tx_buffer);
				if (PIOS_COM_Init(&pios_com_telem_usb_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
							rx_buffer, PIOS_COM_TELEM_USB_RX_BUF_LEN,
							tx_buffer, PIOS_COM_TELEM_USB_TX_BUF_LEN)) {
					PIOS_Assert(0);
				}
			}
#endif  /* PIOS_INCLUDE_COM */
			break;
		case HWSHARED_USB_VCPPORT_COMBRIDGE:
#if defined(PIOS_INCLUDE_COM)
			{
				uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_BRIDGE_RX_BUF_LEN);
				uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_BRIDGE_TX_BUF_LEN);
				PIOS_Assert(rx_buffer);
				PIOS_Assert(tx_buffer);
				if (PIOS_COM_Init(&pios_com_vcp_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
							rx_buffer, PIOS_COM_BRIDGE_RX_BUF_LEN,
							tx_buffer, PIOS_COM_BRIDGE_TX_BUF_LEN)) {
					PIOS_Assert(0);
				}
			}
#endif  /* PIOS_INCLUDE_COM */
			break;
		case HWSHARED_USB_VCPPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_COM)
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			{
				uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN);
				PIOS_Assert(tx_buffer);
				if (PIOS_COM_Init(&pios_com_debug_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
							NULL, 0,
							tx_buffer, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN)) {
					PIOS_Assert(0);
				}
			}
#endif  /* PIOS_INCLUDE_DEBUG_CONSOLE */
#endif  /* PIOS_INCLUDE_COM */
			break;
		case HWSHARED_USB_VCPPORT_PICOC:
#if defined(PIOS_INCLUDE_COM)
			{
				uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_PICOC_RX_BUF_LEN);
				uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_PICOC_TX_BUF_LEN);
				PIOS_Assert(rx_buffer);
				PIOS_Assert(tx_buffer);
				if (PIOS_COM_Init(&pios_com_picoc_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
							rx_buffer, PIOS_COM_PICOC_RX_BUF_LEN,
							tx_buffer, PIOS_COM_PICOC_TX_BUF_LEN)) {
					PIOS_Assert(0);
				}
			}
#endif  /* PIOS_INCLUDE_COM */
			break;
	}
}
#endif

#if defined(PIOS_INCLUDE_USB_HID)
void PIOS_HAL_configure_HID(HwSharedUSB_HIDPortOptions port_type,
		uintptr_t usb_id,
		const struct pios_usb_hid_cfg *hid_cfg) {
	uintptr_t pios_usb_hid_id;
	if (PIOS_USB_HID_Init(&pios_usb_hid_id, hid_cfg, usb_id)) {
		PIOS_Assert(0);
	}

	switch (port_type) {
		case HWSHARED_USB_HIDPORT_DISABLED:
			break;
		case HWSHARED_USB_HIDPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
			{
				uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
				uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
				PIOS_Assert(rx_buffer);
				PIOS_Assert(tx_buffer);
				if (PIOS_COM_Init(&pios_com_telem_usb_id, &pios_usb_hid_com_driver, pios_usb_hid_id,
							rx_buffer, PIOS_COM_TELEM_USB_RX_BUF_LEN,
							tx_buffer, PIOS_COM_TELEM_USB_TX_BUF_LEN)) {
					PIOS_Assert(0);
				}
			}
#endif  /* PIOS_INCLUDE_COM */
			break;
		case HWSHARED_USB_HIDPORT_RCTRANSMITTER:
#if defined(PIOS_INCLUDE_USB_RCTX)
			{
				if (PIOS_USB_RCTX_Init(&pios_usb_rctx_id, &pios_usb_rctx_cfg, pios_usb_id)) {
					PIOS_Assert(0);
				}
			}
#endif  /* PIOS_INCLUDE_USB_RCTX */
			break;

	}
}
#endif  /* PIOS_INCLUDE_USB_HID */


#if defined(PIOS_INCLUDE_RFM22B)
void PIOS_HAL_configure_RFM22B(HwSharedRadioPortOptions radio_type,
		uint8_t board_type, uint8_t board_rev,
		HwSharedMaxRfPowerOptions max_power,
		HwSharedMaxRfSpeedOptions max_speed,
		const struct pios_openlrs_cfg *openlrs_cfg,
		const struct pios_rfm22b_cfg *rfm22b_cfg,
		uint8_t min_chan, uint8_t max_chan, uint32_t coord_id) {
	/* Initalize the RFM22B radio COM device. */
	RFM22BStatusInitialize();
	RFM22BStatusCreateInstance();

	RFM22BStatusData rfm22bstatus;
	RFM22BStatusGet(&rfm22bstatus);
	RFM22BStatusInstSet(1,&rfm22bstatus);

	// Initialize out status object.
	rfm22bstatus.BoardType     = board_type;
	rfm22bstatus.BoardRevision = board_rev;

	if (radio_type == HWSHARED_RADIOPORT_OPENLRS) {
		uintptr_t openlrs_id;

		PIOS_OpenLRS_Init(&openlrs_id, PIOS_RFM22_SPI_PORT, 0, openlrs_cfg);

#if defined(PIOS_INCLUDE_OPENLRS_RCVR)
		{
			uintptr_t pios_rfm22brcvr_id;
			PIOS_OpenLRS_Rcvr_Init(&pios_rfm22brcvr_id, openlrs_id);
			uintptr_t pios_rfm22brcvr_rcvr_id;
			if (PIOS_RCVR_Init(&pios_rfm22brcvr_rcvr_id, &pios_openlrs_rcvr_driver, pios_rfm22brcvr_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_OPENLRS] = pios_rfm22brcvr_rcvr_id;
		}
#endif /* PIOS_INCLUDE_OPENLRS_RCVR */
	} else if (radio_type == HWSHARED_RADIOPORT_DISABLED ||
			max_power == HWSHARED_MAXRFPOWER_0) {
		// When radio disabled, it is ok for init to fail. This allows
		// boards without populating this component.
		if (PIOS_RFM22B_Init(&pios_rfm22b_id, PIOS_RFM22_SPI_PORT, rfm22b_cfg->slave_num, rfm22b_cfg) == 0) {
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_0);
			rfm22bstatus.DeviceID = PIOS_RFM22B_DeviceID(pios_rfm22b_id);
			rfm22bstatus.BoardRevision = PIOS_RFM22B_ModuleVersion(pios_rfm22b_id);
		} else {
			pios_rfm22b_id = 0;
		}

		rfm22bstatus.LinkState = RFM22BSTATUS_LINKSTATE_DISABLED;
	} else {
		// always allow receiving PPM when radio is on
		bool ppm_mode    = radio_type == HWSHARED_RADIOPORT_TELEMPPM || 
			radio_type == HWSHARED_RADIOPORT_PPM;
		bool ppm_only    = radio_type == HWSHARED_RADIOPORT_PPM;
		bool is_oneway   = radio_type == HWSHARED_RADIOPORT_PPM;
			// Sparky2 can only receive PPM only

		/* Configure the RFM22B device. */
		if (PIOS_RFM22B_Init(&pios_rfm22b_id, PIOS_RFM22_SPI_PORT, rfm22b_cfg->slave_num, rfm22b_cfg)) {
			PIOS_Assert(0);
		}

		rfm22bstatus.DeviceID = PIOS_RFM22B_DeviceID(pios_rfm22b_id);
		rfm22bstatus.BoardRevision = PIOS_RFM22B_ModuleVersion(pios_rfm22b_id);

		/* Configure the radio com interface */
		uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_RFM22B_RF_RX_BUF_LEN);
		uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_RFM22B_RF_TX_BUF_LEN);

		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_rf_id, &pios_rfm22b_com_driver, pios_rfm22b_id,
					rx_buffer, PIOS_COM_RFM22B_RF_RX_BUF_LEN,
					tx_buffer, PIOS_COM_RFM22B_RF_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}

		/* Set Telemetry to use RFM22b if no other telemetry is configured (USB always overrides anyway) */
		if (!pios_com_telem_rf_id) {
			pios_com_telem_rf_id = pios_com_rf_id;
		}
		rfm22bstatus.LinkState = RFM22BSTATUS_LINKSTATE_ENABLED;
		
		// XXX TODO: Factor these datarate and power switches
		// out.
		enum rfm22b_datarate datarate;
		switch (max_speed) {
			case HWSHARED_MAXRFSPEED_9600:
				datarate = RFM22_datarate_9600;
				break;
			case HWSHARED_MAXRFSPEED_19200:
				datarate = RFM22_datarate_19200;
				break;
			case HWSHARED_MAXRFSPEED_32000:
				datarate = RFM22_datarate_32000;
				break;
			default:
			case HWSHARED_MAXRFSPEED_64000:
				datarate = RFM22_datarate_64000;
				break;
			case HWSHARED_MAXRFSPEED_100000:
				datarate = RFM22_datarate_100000;
				break;
			case HWSHARED_MAXRFSPEED_192000:
				datarate = RFM22_datarate_192000;
				break;
		}

		/* Set the radio configuration parameters. */
		PIOS_RFM22B_Config(pios_rfm22b_id, datarate, min_chan, max_chan, coord_id, is_oneway, ppm_mode, ppm_only);

		/* Set the modem Tx poer level */
		switch (max_power) {
			case HWSHARED_MAXRFPOWER_125:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_0);
				break;
			case HWSHARED_MAXRFPOWER_16:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_1);
				break;
			case HWSHARED_MAXRFPOWER_316:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_2);
				break;
			case HWSHARED_MAXRFPOWER_63:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_3);
				break;
			case HWSHARED_MAXRFPOWER_126:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_4);
				break;
			case HWSHARED_MAXRFPOWER_25:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_5);
				break;
			case HWSHARED_MAXRFPOWER_50:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_6);
				break;
			case HWSHARED_MAXRFPOWER_100:
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_7);
				break;
			default:
				// do nothing
				break;
		}

		/* Reinitialize the modem. */
		PIOS_RFM22B_Reinit(pios_rfm22b_id);

#if defined(PIOS_INCLUDE_RFM22B_RCVR)
		{
			uintptr_t pios_rfm22brcvr_id;
			PIOS_RFM22B_Rcvr_Init(&pios_rfm22brcvr_id, pios_rfm22b_id);
			uintptr_t pios_rfm22brcvr_rcvr_id;
			if (PIOS_RCVR_Init(&pios_rfm22brcvr_rcvr_id, &pios_rfm22b_rcvr_driver, pios_rfm22brcvr_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_RFM22B] = pios_rfm22brcvr_rcvr_id;
		}
#endif /* PIOS_INCLUDE_RFM22B_RCVR */
	}

	RFM22BStatusInstSet(1,&rfm22bstatus);
}
#endif /* PIOS_INCLUDE_RFM22B */
