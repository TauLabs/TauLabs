/**
 ******************************************************************************
 * @addtogroup TauLabsBootloader Tau Labs Bootloaders
 * @{
 * @addtogroup CopterControlBL CopterControl bootloader
 * @{
 *
 * @file       op_dfu.c 
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      This file contains the DFU commands handling code
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

/* Includes ------------------------------------------------------------------*/
#include "pios.h"
#include "op_dfu.h"
#include "pios_bl_helper.h"
#include "pios_com_msg.h"
#include <pios_board_info.h>
//programmable devices
Device devicesTable[10];
uint8_t numberOfDevices = 0;

DFUProgType currentProgrammingDestination; //flash, flash_trough spi
uint8_t currentDeviceCanRead;
uint8_t currentDeviceCanWrite;
Device currentDevice;

uint8_t Buffer[64];
uint8_t echoBuffer[64];
uint8_t SendBuffer[64];
uint8_t Command = 0;
uint8_t EchoReqFlag = 0;
uint8_t EchoAnsFlag = 0;
uint8_t StartFlag = 0;
uint32_t Aditionals = 0;
uint32_t SizeOfTransfer = 0;
uint32_t Expected_CRC = 0;
uint8_t SizeOfLastPacket = 0;
uint32_t Next_Packet = 0;
uint8_t TransferType;
uint32_t Count = 0;
uint32_t Data;
uint8_t Data0;
uint8_t Data1;
uint8_t Data2;
uint8_t Data3;
uint8_t offset = 0;
uint32_t aux;
//Download vars
uint32_t downSizeOfLastPacket = 0;
uint32_t downPacketTotal = 0;
uint32_t downPacketCurrent = 0;
DFUTransfer downType = 0;
/* Extern variables ----------------------------------------------------------*/
extern DFUStates DeviceState;
extern uint8_t JumpToApp;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
void sendData(uint8_t * buf, uint16_t size);
uint32_t CalcFirmCRC(void);

void DataDownload(DownloadAction action) {
	if ((DeviceState == downloading)) {

		uint8_t packetSize;
		uint32_t offset;
		uint32_t partoffset;
		SendBuffer[0] = 0x01;
		SendBuffer[1] = Download;
		SendBuffer[2] = downPacketCurrent >> 24;
		SendBuffer[3] = downPacketCurrent >> 16;
		SendBuffer[4] = downPacketCurrent >> 8;
		SendBuffer[5] = downPacketCurrent;
		if (downPacketCurrent == downPacketTotal - 1) {
			packetSize = downSizeOfLastPacket;
		} else {
			packetSize = 14;
		}
		for (uint8_t x = 0; x < packetSize; ++x) {
			partoffset = (downPacketCurrent * 14 * 4) + (x * 4);
			offset = baseOfAdressType(downType) + partoffset;
			if (!flash_read(SendBuffer + (6 + x * 4), offset,
					currentProgrammingDestination)) {
				DeviceState = Last_operation_failed;
			}
		}
		downPacketCurrent = downPacketCurrent + 1;
		if (downPacketCurrent > downPacketTotal - 1) {
			DeviceState = Last_operation_Success;
			Aditionals = (uint32_t) Download;
		}
		sendData(SendBuffer + 1, 63);
	}
}
void processComand(uint8_t *xReceive_Buffer) {

	Command = xReceive_Buffer[COMMAND];
#ifdef DEBUG_SSP
	char str[63]= {0};
	sprintf(str,"Received COMMAND:%d|",Command);
	PIOS_COM_SendString(PIOS_COM_TELEM_USB,str);
#endif
	EchoReqFlag = (Command >> 7);
	EchoAnsFlag = (Command >> 6) & 0x01;
	StartFlag = (Command >> 5) & 0x01;
	Count = xReceive_Buffer[COUNT] << 24;
	Count += xReceive_Buffer[COUNT + 1] << 16;
	Count += xReceive_Buffer[COUNT + 2] << 8;
	Count += xReceive_Buffer[COUNT + 3];

	Data = xReceive_Buffer[DATA] << 24;
	Data += xReceive_Buffer[DATA + 1] << 16;
	Data += xReceive_Buffer[DATA + 2] << 8;
	Data += xReceive_Buffer[DATA + 3];
	Data0 = xReceive_Buffer[DATA];
	Data1 = xReceive_Buffer[DATA + 1];
	Data2 = xReceive_Buffer[DATA + 2];
	Data3 = xReceive_Buffer[DATA + 3];
	Command = Command & 0b00011111;

	if (EchoReqFlag == 1) {
		memcpy(echoBuffer, Buffer, 64);
	}
	switch (Command) {
	case EnterDFU:
		if (((DeviceState == BLidle) && (Data0 < numberOfDevices))
				|| (DeviceState == DFUidle)) {
			if (Data0 > 0)
				OPDfuIni(true);
			DeviceState = DFUidle;
			currentProgrammingDestination = devicesTable[Data0].programmingType;
			currentDeviceCanRead = devicesTable[Data0].readWriteFlags & 0x01;
			currentDeviceCanWrite = devicesTable[Data0].readWriteFlags >> 1
					& 0x01;
			currentDevice = devicesTable[Data0];
			uint8_t result = 0;
			switch (currentProgrammingDestination) {
			case Self_flash:
				result = PIOS_BL_HELPER_FLASH_Ini();
				break;
			case Remote_flash_via_spi:
				result = true;
				break;
			default:
				DeviceState = Last_operation_failed;
				Aditionals = (uint16_t) Command;
			}
			if (result != 1) {
				DeviceState = Last_operation_failed;
				Aditionals = (uint32_t) Command;
			}
		}
		break;
	case Upload:
		if ((DeviceState == DFUidle) || (DeviceState == uploading)) {
			if ((StartFlag == 1) && (Next_Packet == 0)) {
				TransferType = Data0;
				SizeOfTransfer = Count;
				Next_Packet = 1;
				Expected_CRC = Data2 << 24;
				Expected_CRC += Data3 << 16;
				Expected_CRC += xReceive_Buffer[DATA + 4] << 8;
				Expected_CRC += xReceive_Buffer[DATA + 5];
				SizeOfLastPacket = Data1;

				if (isBiggerThanAvailable(TransferType, (SizeOfTransfer - 1)
						* 14 * 4 + SizeOfLastPacket * 4) == true) {
					DeviceState = outsideDevCapabilities;
					Aditionals = (uint32_t) Command;
				} else {
					uint8_t result = 1;
					if (TransferType == FW) {
						switch (currentProgrammingDestination) {
						case Self_flash:
							result = PIOS_BL_HELPER_FLASH_Start();
							break;
						case Remote_flash_via_spi:
							result = false;
							break;
						default:
							break;
						}
					}
					if (result != 1) {
						DeviceState = Last_operation_failed;
						Aditionals = (uint32_t) Command;
					} else {

						DeviceState = uploading;
					}
				}
			} else if ((StartFlag != 1) && (Next_Packet != 0)) {
				if (Count > SizeOfTransfer) {
					DeviceState = too_many_packets;
					Aditionals = Count;
				} else if (Count == Next_Packet - 1) {
					uint8_t numberOfWords = 14;
					if (Count == SizeOfTransfer - 1)//is this the last packet?
					{
						numberOfWords = SizeOfLastPacket;
					}
					uint8_t result = 0;
					switch (currentProgrammingDestination) {
					case Self_flash:
						for (uint8_t x = 0; x < numberOfWords; ++x) {
							offset = 4 * x;
							Data = xReceive_Buffer[DATA + offset] << 24;
							Data += xReceive_Buffer[DATA + 1 + offset] << 16;
							Data += xReceive_Buffer[DATA + 2 + offset] << 8;
							Data += xReceive_Buffer[DATA + 3 + offset];
							aux = baseOfAdressType(TransferType) + (uint32_t)(
									Count * 14 * 4 + x * 4);
							result = 0;
							for (int retry = 0; retry < MAX_WRI_RETRYS; ++retry) {
								if (result == 0) {
									result = (FLASH_ProgramWord(aux, Data)
											== FLASH_COMPLETE) ? 1 : 0;
								}
							}
						}
						break;
					case Remote_flash_via_spi:
						result = false; // No support for this for the PipX
						break;
					default:
						result = 0;
						break;
					}
					if (result != 1) {
						DeviceState = Last_operation_failed;
						Aditionals = (uint32_t) Command;
					}

					++Next_Packet;
				} else {
					DeviceState = wrong_packet_received;
					Aditionals = Count;
				}
			} else {
				DeviceState = Last_operation_failed;
				Aditionals = (uint32_t) Command;
			}
		}
		break;
	case Req_Capabilities:
		OPDfuIni(true);
		Buffer[0] = 0x01;
		Buffer[1] = Rep_Capabilities;
		if (Data0 == 0) {
			Buffer[2] = 0;
			Buffer[3] = 0;
			Buffer[4] = 0;
			Buffer[5] = 0;
			Buffer[6] = 0;
			Buffer[7] = numberOfDevices;
			uint16_t WRFlags = 0;
			for (int x = 0; x < numberOfDevices; ++x) {
				WRFlags = ((devicesTable[x].readWriteFlags << (x * 2))
						| WRFlags);
			}
			Buffer[8] = WRFlags >> 8;
			Buffer[9] = WRFlags;
		} else {
			Buffer[2] = devicesTable[Data0 - 1].sizeOfCode >> 24;
			Buffer[3] = devicesTable[Data0 - 1].sizeOfCode >> 16;
			Buffer[4] = devicesTable[Data0 - 1].sizeOfCode >> 8;
			Buffer[5] = devicesTable[Data0 - 1].sizeOfCode;
			Buffer[6] = Data0;
			Buffer[7] = devicesTable[Data0 - 1].BL_Version;
			Buffer[8] = devicesTable[Data0 - 1].sizeOfDescription;
			Buffer[9] = devicesTable[Data0 - 1].devID;
			Buffer[10] = devicesTable[Data0 - 1].FW_Crc >> 24;
			Buffer[11] = devicesTable[Data0 - 1].FW_Crc >> 16;
			Buffer[12] = devicesTable[Data0 - 1].FW_Crc >> 8;
			Buffer[13] = devicesTable[Data0 - 1].FW_Crc;
			Buffer[14] = devicesTable[Data0 - 1].devID >> 8;
			Buffer[15] = devicesTable[Data0 - 1].devID;
		}
		sendData(Buffer + 1, 63);
		break;
	case JumpFW:
		if (Data == 0x5AFE) {
			/* Force board into safe mode */
			PIOS_IAP_WriteBootCount(0xFFFF);
		}
		FLASH_Lock();
		JumpToApp = 1;
		break;
	case Reset:
		PIOS_SYS_Reset();
		break;
	case Abort_Operation:
		Next_Packet = 0;
		DeviceState = DFUidle;
		break;

	case Op_END:
		if (DeviceState == uploading) {
			if (Next_Packet - 1 == SizeOfTransfer) {
				Next_Packet = 0;
				if ((TransferType != FW) || (Expected_CRC == CalcFirmCRC())) {
					DeviceState = Last_operation_Success;
				} else {
					DeviceState = CRC_Fail;
				}
			}
			if (Next_Packet - 1 < SizeOfTransfer) {
				Next_Packet = 0;
				DeviceState = too_few_packets;
			}
		}
		break;
	case Download_Req:
#ifdef DEBUG_SSP
		sprintf(str,"COMMAND:DOWNLOAD_REQ 1 Status=%d|",DeviceState);
		PIOS_COM_SendString(PIOS_COM_TELEM_USB,str);
#endif
		if (DeviceState == DFUidle) {
#ifdef DEBUG_SSP
			PIOS_COM_SendString(PIOS_COM_TELEM_USB,"COMMAND:DOWNLOAD_REQ 1|");
#endif
			downType = Data0;
			downPacketTotal = Count;
			downSizeOfLastPacket = Data1;
			if (isBiggerThanAvailable(downType, (downPacketTotal - 1) * 14
					+ downSizeOfLastPacket) == 1) {
				DeviceState = outsideDevCapabilities;
				Aditionals = (uint32_t) Command;

			} else {
				downPacketCurrent = 0;
				DeviceState = downloading;
			}
		} else {
			DeviceState = Last_operation_failed;
			Aditionals = (uint32_t) Command;
		}
		break;

	case Status_Request:
		Buffer[0] = 0x01;
		Buffer[1] = Status_Rep;
		if (DeviceState == wrong_packet_received) {
			Buffer[2] = Aditionals >> 24;
			Buffer[3] = Aditionals >> 16;
			Buffer[4] = Aditionals >> 8;
			Buffer[5] = Aditionals;
		} else {
			Buffer[2] = 0;
			Buffer[3] = ((uint16_t) Aditionals) >> 8;
			Buffer[4] = ((uint16_t) Aditionals);
			Buffer[5] = 0;
		}
		Buffer[6] = DeviceState;
		Buffer[7] = 0;
		Buffer[8] = 0;
		Buffer[9] = 0;
		sendData(Buffer + 1, 63);
		if (DeviceState == Last_operation_Success) {
			DeviceState = DFUidle;
		}
		break;
	case Status_Rep:

		break;

	}
	if (EchoReqFlag == 1) {
		echoBuffer[1] = echoBuffer[1] | EchoAnsFlag;
		sendData(echoBuffer + 1, 63);
	}
	return;
}
void OPDfuIni(uint8_t discover) {
	const struct pios_board_info * bdinfo = &pios_board_info_blob;
	Device dev;

	dev.programmingType = Self_flash;
	dev.readWriteFlags = (BOARD_READABLE | (BOARD_WRITABLE << 1));
	dev.startOfUserCode = bdinfo->fw_base;
	dev.sizeOfCode = bdinfo->fw_size;
	dev.sizeOfDescription = bdinfo->desc_size;
	dev.BL_Version = bdinfo->bl_rev;
	dev.FW_Crc = CalcFirmCRC();
	dev.devID = (bdinfo->board_type << 8) | (bdinfo->board_rev);
	dev.devType = bdinfo->hw_type;
	numberOfDevices = 1;
	devicesTable[0] = dev;
	if (discover) {
		//TODO check other devices trough spi or whatever
	}
}
uint32_t baseOfAdressType(DFUTransfer type) {
	switch (type) {
	case FW:
		return currentDevice.startOfUserCode;
		break;
	case Descript:
		return currentDevice.startOfUserCode + currentDevice.sizeOfCode;
		break;
	default:

		return 0;
	}
}
uint8_t isBiggerThanAvailable(DFUTransfer type, uint32_t size) {
	switch (type) {
	case FW:
		return (size > currentDevice.sizeOfCode) ? 1 : 0;
		break;
	case Descript:
		return (size > currentDevice.sizeOfDescription) ? 1 : 0;
		break;
	default:
		return true;
	}
}

uint32_t CalcFirmCRC() {
	switch (currentProgrammingDestination) {
	case Self_flash:
		return PIOS_BL_HELPER_CRC_Memory_Calc();
		break;
	case Remote_flash_via_spi:
		return 0;
		break;
	default:
		return 0;
		break;
	}

}
void sendData(uint8_t * buf, uint16_t size) {
	PIOS_COM_MSG_Send(PIOS_COM_TELEM_USB, buf, size);
}

bool flash_read(uint8_t * buffer, uint32_t adr, DFUProgType type) {
	switch (type) {
	case Remote_flash_via_spi:
		return false; // We should not get this for the PipX
		break;
	case Self_flash:
		for (uint8_t x = 0; x < 4; ++x) {
			buffer[x] = *PIOS_BL_HELPER_FLASH_If_Read(adr + x);
		}
		return true;
		break;
	default:
		return false;
	}
}
