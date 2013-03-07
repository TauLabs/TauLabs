/******************** (C) COPYRIGHT 2009 STMicroelectronics ********************
* File Name          : usb_desc.c
* Author             : MCD Application Team
* Version            : V3.0.1
* Date               : 04/27/2009
* Description        : Descriptors for Mass Storage Device
********************************************************************************
* THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
* WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE TIME.
* AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY DIRECT,
* INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING FROM THE
* CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE CODING
* INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
*******************************************************************************/

/* Includes ------------------------------------------------------------------*/
#include "msd_desc.h"

const uint8_t MSD_MASS_DeviceDescriptor[MSD_MASS_SIZ_DEVICE_DESC] =
  {
    0x12,   /* bLength  */
    0x01,   /* bDescriptorType */
    0x00,   /* bcdUSB, version 2.00 */
    0x02,
    0x00,   /* bDeviceClass : each interface define the device class */
    0x00,   /* bDeviceSubClass */
    0x00,   /* bDeviceProtocol */
    0x40,   /* bMaxPacketSize0 0x40 = 64 */
    0x83,   /* idVendor     (0483) */
    0x04,
    0x20,   /* idProduct */
    0x57,
    0x00,   /* bcdDevice 2.00*/
    0x02,
    1,              /* index of string Manufacturer  */
    /**/
    2,              /* index of string descriptor of product*/
    /* */
    3,              /* */
    /* */
    /* */
    0x01    /*bNumConfigurations */
  };
const uint8_t MSD_MASS_ConfigDescriptor[MSD_MASS_SIZ_CONFIG_DESC] =
  {

    0x09,   /* bLength: Configuation Descriptor size */
    0x02,   /* bDescriptorType: Configuration */
    MSD_MASS_SIZ_CONFIG_DESC,

    0x00,
    0x01,   /* bNumInterfaces: 1 interface */
    0x01,   /* bConfigurationValue: */
    /*      Configuration value */
    0x00,   /* iConfiguration: */
    /*      Index of string descriptor */
    /*      describing the configuration */
    0xC0,   /* bmAttributes: */
    /*      bus powered */
    0x32,   /* MaxPower 100 mA */

    /******************** Descriptor of Mass Storage interface ********************/
    /* 09 */
    0x09,   /* bLength: Interface Descriptor size */
    0x04,   /* bDescriptorType: */
    /*      Interface descriptor type */
    0x00,   /* bInterfaceNumber: Number of Interface */
    0x00,   /* bAlternateSetting: Alternate setting */
    0x02,   /* bNumEndpoints*/
    0x08,   /* bInterfaceClass: MASS STORAGE Class */
    0x06,   /* bInterfaceSubClass : SCSI transparent*/
    0x50,   /* nInterfaceProtocol */
    4,          /* iInterface: */
    /* 18 */
    0x07,   /*Endpoint descriptor length = 7*/
    0x05,   /*Endpoint descriptor type */
    0x81,   /*Endpoint address (IN, address 1) */
    0x02,   /*Bulk endpoint type */
    0x40,   /*Maximum packet size (64 bytes) */
    0x00,
    0x00,   /*Polling interval in milliseconds */
    /* 25 */
    0x07,   /*Endpoint descriptor length = 7 */
    0x05,   /*Endpoint descriptor type */
    0x02,   /*Endpoint address (OUT, address 2) */
    0x02,   /*Bulk endpoint type */
    0x40,   /*Maximum packet size (64 bytes) */
    0x00,
    0x00     /*Polling interval in milliseconds*/
    /*32*/
  };
const uint8_t MSD_MASS_StringLangID[MSD_MASS_SIZ_STRING_LANGID] =
  {
    MSD_MASS_SIZ_STRING_LANGID,
    0x03,
    0x09,
    0x04
  }
  ;      /* LangID = 0x0409: U.S. English */
const uint8_t MSD_MASS_StringVendor[MSD_MASS_SIZ_STRING_VENDOR] =
  {
    MSD_MASS_SIZ_STRING_VENDOR, /* Size of manufaturer string */
    0x03,           /* bDescriptorType = String descriptor */
    /* Manufacturer */
    'T', 0, 'a', 0, 'u', 0, ' ', 0, 'L', 0, 'a', 0, 'b', 0, 's', 0,
  };
const uint8_t MSD_MASS_StringProduct[MSD_MASS_SIZ_STRING_PRODUCT] =
  {
    MSD_MASS_SIZ_STRING_PRODUCT,
    0x03,
    /* Product name */
    'T', 0, 'a', 0, 'u', 0, ' ', 0, 'L', 0, 'a', 0, 'b', 0, 's', 0,
    ' ', 0, 'M', 0, 'a', 0, 's', 0, 's', 0, ' ', 0, 'S', 0, 't', 0, 'o', 0,
    'r', 0, 'a', 0, 'g', 0, 'e', 0

  };

uint8_t MSD_MASS_StringSerial[MSD_MASS_SIZ_STRING_SERIAL] =
  {
    MSD_MASS_SIZ_STRING_SERIAL,
    0x03,
    /* Serial number */
    '0', 0, '0', 0, '0', 0, '0', 0, '0', 0, '0', 0, '0', 0
  };
const uint8_t MSD_MASS_StringInterface[MSD_MASS_SIZ_STRING_INTERFACE] =
  {
    MSD_MASS_SIZ_STRING_INTERFACE,
    0x03,
    /* Interface 0: */
    'S', 0, 'D', 0, ' ', 0, 'C', 0, 'a', 0, 'r', 0, 'd', 0
  };

/******************* (C) COPYRIGHT 2009 STMicroelectronics *****END OF FILE****/

