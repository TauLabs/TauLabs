/**
 ******************************************************************************
 *
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_FLASH Flash device handler
 * @{
 *
 * @file       pios_AT45_flash_jedec.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Virtual Robotix Network Team, http://www.virtualrobotix.com Copyright (C) 2013.
 * @brief      Driver for talking to AT45DB161D flash chip (and most JEDEC chips)
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

#define AT45DB_BUFFER_1_WRITE                 0x84	/* write the first buffer */
#define AT45DB_BUFFER_2_WRITE                 0x87	/* write the second buffer */
#define AT45DB_BUFFER_1_READ                  0xD4	/* read the first buffer */
#define AT45DB_BUFFER_2_READ                  0xD6	/* read the second buffer */
#define AT45DB_B1_TO_MM_PAGE_PROG_WITH_ERASE  0x83	/* first buffer data is written to main memory (erase mode) */
#define AT45DB_B2_TO_MM_PAGE_PROG_WITH_ERASE  0x86	/* second buffer data is written to the main memory (erase mode) */
#define AT45DB_MM_PAGE_TO_B1_XFER             0x53	/* specified page of the main memory data is loaded into the first buffer */
#define AT45DB_MM_PAGE_TO_B2_XFER             0x55	/* load the specified page data of the main memory to the second buffer */
#define AT45DB_PAGE_ERASE                     0x81	/* delete page (page 512/528 bytes) */
#define AT45DB_SECTOR_ERASE                   0x7C	/* sector erase (128K bytes per sector) */
#define AT45DB_READ_STATE_REGISTER            0xD7	/* read status register */
#define AT45DB_MM_PAGE_READ                   0xD2	/* read directly the main memory of the memory page */
#define AT45DB_MM_PAGE_PROG_THRU_BUFFER1      0x82	/* write the first buffer */

struct pios_flash_jedec_cfg {
	uint32_t sector_erase;
	uint32_t chip_erase;
};

struct pios_flash_chunk {
	uint8_t * addr;
	uint32_t len;
};

//uint32_t df_PageSize;

int32_t PIOS_Flash_Jedec_Init(uint32_t spi_id, uint32_t slave_num, const struct pios_flash_jedec_cfg * cfg);
int32_t PIOS_Flash_Jedec_ReadStatus();
int32_t PIOS_Flash_Jedec_ReadID();
int32_t PIOS_Flash_Jedec_EraseChip();
int32_t PIOS_Flash_Jedec_ErasePage(uint32_t add);
int32_t PIOS_Flash_Jedec_WriteData(uint32_t addr, uint8_t * data, uint16_t len);
int32_t PIOS_Flash_Jedec_ReadData(uint32_t addr, uint8_t * data, uint16_t len);
int32_t PIOS_Flash_Jedec_WriteChunks(uint32_t addr, struct pios_flash_chunk * p_chunk, uint32_t num);
int32_t PIOS_Flash_Jedec_StartTransaction();
int32_t PIOS_Flash_Jedec_EndTransaction();
int32_t PIOS_Flash_Jedec_PageSize();
int32_t PIOS_Flash_Jedec_PageToBuffer(unsigned char BufferNum, uint16_t PageAdr);
int32_t PIOS_Flash_Jedec_BufferToPage(unsigned char BufferNum, uint16_t PageAdr);
int32_t PIOS_Flash_Jedec_BufferWrite (unsigned char BufferNum, uint16_t IntPageAdr,  uint8_t * data, uint16_t len);
int32_t PIOS_Flash_Jedec_BufferRead (unsigned char BufferNum, uint16_t IntBufferAdr,  uint8_t * data, uint16_t len);
int32_t PIOS_Flash_Jedec_ReadPage(uint32_t page, uint16_t offset, uint8_t * data, uint16_t len);
int32_t PIOS_Flash_Jedec_WritePage(uint32_t page, uint16_t offset, uint8_t * data, uint16_t len);

int32_t PIOS_Flash_Jedec_PrintID();


