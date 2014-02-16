/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picoc_module.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      c-interpreter module for autonomous user programmed tasks
 *             picoc module task
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


// conditional compilation of the module
#include "pios.h"
#ifdef PIOS_INCLUDE_PICOC

#include "openpilot.h"
#include "picoc_port.h"
#include "picocsettings.h" 
#include "picocstatus.h" 
#include "flightstatus.h"
#include "modulesettings.h"

// Global variables
extern uintptr_t pios_waypoints_settings_fs_id;	/* use the waypoint filesystem */
extern struct flashfs_logfs_cfg flashfs_waypoints_cfg;

// Private constants
#define TASK_PRIORITY			(tskIDLE_PRIORITY + 1)
#define TASK_STACKSIZE_MIN		(5*1024)
#define TASK_STACKSIZE_MAX		(128*1024)
#define PICOC_STACKSIZE_MIN		(10*1024)
#define PICOC_STACKSIZE_MAX		(128*1024)
#define PICOC_SOURCE_FILE_TYPE	0X00704300		/* mark picoc sources with this ID */
#define PICOC_SECTOR_SIZE		48				/* size of filesystem object (less than slot_size - sizeof(slot_header) */
#define SOH	0x01	/* (^A) start of heading */
#define STX	0x02	/* (^B) start of text */
#define ETX	0x03	/* (^C) end of text */
#define EOT	0x04	/* (^D) end of transmission */
#define ENQ	0x05	/* (^E) enquiry */

// Private variables
static xTaskHandle picocTaskHandle;
static uintptr_t picocPort;
static bool module_enabled;
static char *sourcebuffer;
static uint32_t sourcebuffer_size;
static PicoCSettingsData picocsettings;
static PicoCStatusData picocstatus;

// Private functions
static void picocTask(void *parameters);
static void updateSettings();
int32_t usart_cmd(char *buffer, uint32_t buffer_size);
int32_t get_sector(uint16_t sector, char *buffer, uint32_t buffer_size);
int32_t set_sector(uint16_t sector, char *buffer, uint32_t buffer_size);
int32_t load_file(uint8_t file, char *buffer, uint32_t buffer_size);
int32_t save_file(uint8_t file, char *buffer, uint32_t buffer_size);
int32_t delete_file(uint8_t file);
int32_t format_partition();

/**
 * start the module
 * \return -1 if start failed
 * \return 0 on success
 */
static int32_t picocStart(void)
{
	if (module_enabled) {
		// Start task
		xTaskCreate(picocTask, (signed char *) "PicoC",
				picocsettings.TaskStackSize / 4, NULL, TASK_PRIORITY,
				&picocTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PICOC,
				picocTaskHandle);
		return 0;
	}
	return -1;
}

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t picocInitialize(void)
{
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	module_enabled = false;

	if (module_state[MODULESETTINGS_ADMINSTATE_PICOC] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		PicoCSettingsInitialize();
		PicoCStatusInitialize();

		// get picoc settings
		PicoCSettingsGet(&picocsettings);

		// check stacksizes for module task and picoC
		if ((picocsettings.TaskStackSize < TASK_STACKSIZE_MIN) || (picocsettings.TaskStackSize > TASK_STACKSIZE_MAX) ||
			(picocsettings.PicoCStackSize < PICOC_STACKSIZE_MIN) || (picocsettings.PicoCStackSize > PICOC_STACKSIZE_MAX)) {
			return -1;
		}

		// allocate memory for source buffer
		sourcebuffer_size = picocsettings.MaxFileSize;
		if (sourcebuffer_size) {
			sourcebuffer = pvPortMalloc(sourcebuffer_size);
		}
		if (sourcebuffer == NULL) {
			// there is not enough free memory for source file buffer. the module could not run.
			return -1;
		}

#ifdef PIOS_COM_PICOC
		// get picoc USART for stdIO communication
		picocPort = PIOS_COM_PICOC;
#endif

		module_enabled = true;
	}
	return 0;
}
MODULE_INITCALL( picocInitialize, picocStart)

/**
 * Main task. It does not return.
 */
static void picocTask(void *parameters) {

	FlightStatusData flightstatus;
	bool startup;
	bool started = false;

	// demo for picoc
	const char *demo = "void test() {for (int i=0; i<10; i++) printf(\"i=%d\\n\", i); } test();";

	// initial USART setup
	updateSettings();

	// clear source buffer
	memset(sourcebuffer, 0, sourcebuffer_size);

	// load boot file from flash
	PicoCSettingsGet(&picocsettings);
	picocstatus.CommandError = load_file(picocsettings.BootFileID, sourcebuffer, sourcebuffer_size);
	PicoCStatusCommandErrorSet(&picocstatus.CommandError);

	while (1) {
		PicoCSettingsGet(&picocsettings);
		PicoCStatusGet(&picocstatus);
		FlightStatusGet(&flightstatus);

		// handle file and buffer commands
		if (picocstatus.Command != PICOCSTATUS_COMMAND_IDLE) {
			switch (picocstatus.Command) {
			case PICOCSTATUS_COMMAND_USARTMODE:
				// handle commands via USART
				picocstatus.CommandError = usart_cmd(sourcebuffer, sourcebuffer_size);
				if (picocstatus.CommandError) {
					picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				}
				break;
			case PICOCSTATUS_COMMAND_GETSECTOR:
				// copy selected sector from buffer to uavo
				picocstatus.CommandError = get_sector(picocstatus.SectorID, sourcebuffer, sourcebuffer_size);
				PicoCStatusSectorSet(picocstatus.Sector);
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				break;
			case PICOCSTATUS_COMMAND_SETSECTOR:
				// fill buffer from uavo to selected sector
				picocstatus.CommandError = set_sector(picocstatus.SectorID, sourcebuffer, sourcebuffer_size);
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				break;
			case PICOCSTATUS_COMMAND_LOADFILE:
				// fill buffer from flash file
				picocstatus.CommandError = load_file(picocstatus.FileID, sourcebuffer, sourcebuffer_size);
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				break;
			case PICOCSTATUS_COMMAND_SAVEFILE:
				// save buffer to flash file
				picocstatus.CommandError = save_file(picocstatus.FileID, sourcebuffer, sourcebuffer_size);
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				break;
			case PICOCSTATUS_COMMAND_DELETEFILE:
				// delete flash file
				picocstatus.CommandError = delete_file(picocstatus.FileID);
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				break;
			case PICOCSTATUS_COMMAND_FORMATPARTITION:
				// delete all flash files in partition
				picocstatus.CommandError = format_partition();
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
				break;
			default:
				// unkown command
				picocstatus.CommandError = -1;
				picocstatus.Command = PICOCSTATUS_COMMAND_IDLE;
			}
			// transfer return values
			if (picocstatus.Command == PICOCSTATUS_COMMAND_IDLE) {
				PicoCStatusCommandErrorSet(&picocstatus.CommandError);
				PicoCStatusCommandSet(&picocstatus.Command);
			}
		}

		// check startup condition
		if ((picocstatus.Command == PICOCSTATUS_COMMAND_IDLE) && (picocstatus.CommandError == 0)) {
			switch (picocsettings.Startup) {
			case PICOCSETTINGS_STARTUP_DISABLED:
				startup = false;
				break;
			case PICOCSETTINGS_STARTUP_ONBOOT:
				startup = true;
				break;
			case PICOCSETTINGS_STARTUP_WHENARMED:
				startup = (flightstatus.Armed == FLIGHTSTATUS_ARMED_ARMED);
				break;
			default:
				startup = false;
			}
		} else {
			startup = false;
		}

		// start picoc interpreter in selected mode
		if (startup && !started) {
			switch (picocsettings.Source) {
			case PICOCSETTINGS_SOURCE_DEMO:
				// run the demo code.
				picocstatus.ExitValue = picoc(demo, picocsettings.PicoCStackSize);
				started = true;
				break;
			case PICOCSETTINGS_SOURCE_INTERACTIVE:
				// start picoc in interactive mode.
				picocstatus.ExitValue = picoc(NULL, picocsettings.PicoCStackSize);
				break;
			case PICOCSETTINGS_SOURCE_FILE:
				// terminate source for security.
				sourcebuffer[sourcebuffer_size - 1] = 0;
				// start picoc in file mode.
				picocstatus.ExitValue = picoc(sourcebuffer, picocsettings.PicoCStackSize);
				started = true;
				break;
			default:
				picocstatus.ExitValue = 0;
			}
			if (picocsettings.Source != PICOCSETTINGS_SOURCE_DISABLED) {
				PicoCStatusExitValueSet(&picocstatus.ExitValue);
			}
		}
		started &= startup;

		vTaskDelay(10);
	}
}

/**
 * update picoc module settings
 */
static void updateSettings()
{
	// if there is a com port, setup its speed.
	if (picocPort) {
		// set port speed
		switch (picocsettings.ComSpeed) {
		case PICOCSETTINGS_COMSPEED_2400:
			PIOS_COM_ChangeBaud(picocPort, 2400);
			break;
		case PICOCSETTINGS_COMSPEED_4800:
			PIOS_COM_ChangeBaud(picocPort, 4800);
			break;
		case PICOCSETTINGS_COMSPEED_9600:
			PIOS_COM_ChangeBaud(picocPort, 9600);
			break;
		case PICOCSETTINGS_COMSPEED_19200:
			PIOS_COM_ChangeBaud(picocPort, 19200);
			break;
		case PICOCSETTINGS_COMSPEED_38400:
			PIOS_COM_ChangeBaud(picocPort, 38400);
			break;
		case PICOCSETTINGS_COMSPEED_57600:
			PIOS_COM_ChangeBaud(picocPort, 57600);
			break;
		case PICOCSETTINGS_COMSPEED_115200:
			PIOS_COM_ChangeBaud(picocPort, 115200);
			break;
		}
	}
}

/**
 * usart command
 */
int32_t usart_cmd(char *buffer, uint32_t buffer_size)
{
	static uint32_t buffer_pointer = 0;
	static bool buffer_enabled = false;
	uint8_t ch;

	// check for a valid USART first
	if (picocPort == 0) {
		return -1;
	}

	// handle incoming data
	while (PIOS_COM_ReceiveBuffer(picocPort, &ch, sizeof(ch), 0) == sizeof(ch)) {
		switch (ch) {
		case SOH:	// start of heading
			PIOS_COM_SendString(picocPort,"start of heading\n");
			memset(buffer, 0, buffer_size);
			buffer_pointer = 0;
			break;
		case STX:	// start of text
			PIOS_COM_SendString(picocPort,"start of text\n");
			buffer[buffer_pointer] = '\0';
			buffer_enabled = true;
			break;
		case ETX:	// end of text
			PIOS_COM_SendString(picocPort,"end of text\n");
			buffer_enabled = false;
			break;
		case EOT:	// end of transmission
			PIOS_COM_SendString(picocPort,"end of transmission\n");
			buffer_pointer = 0;
			buffer_enabled = false;
			break;
		case ENQ:	// enquiry
			// used to readout buffer
			for (int32_t i = 0; ((i < buffer_size) && (buffer[i] != 0)); i++) {
				PIOS_COM_SendChar(picocPort, buffer[i]);
			}
			break;
		default:
			// write incoming data to buffer, if enabled
			if ((buffer_enabled) && (ch != '\0')) {
				if (buffer_pointer >= buffer_size - 2) {
					PIOS_COM_SendString(picocPort,"buffer overrun\n");
					buffer_enabled = false;
				} else {
					buffer[buffer_pointer++] = ch;
					buffer[buffer_pointer] = '\0';
				}
			}
		}
	}

	return 0;
}

/**
 * get sector from source buffer
 */
int32_t get_sector(uint16_t sector, char *buffer, uint32_t buffer_size)
{
	// calculate buffer offset
	uint32_t offset = sector * sizeof(picocstatus.Sector);

	// copy buffer to sector
	for (uint32_t i = 0; i < sizeof(picocstatus.Sector); i++) {
		picocstatus.Sector[i] = (buffer_size > offset + i) ? buffer[offset + i] : 0;
	}

	PicoCStatusSectorSet(picocstatus.Sector);
	return 0;
}

/**
 * put sector to source buffer
 */
int32_t set_sector(uint16_t sector, char *buffer, uint32_t buffer_size)
{
	// calculate buffer offset
	uint32_t offset = sector * sizeof(picocstatus.Sector);

	PicoCStatusSectorGet(picocstatus.Sector);

	// copy sector to buffer
	for (uint32_t i = 0; i < sizeof(picocstatus.Sector); i++) {
		if (buffer_size > offset + i) {
			buffer[offset + i] = picocstatus.Sector[i];
		}
	}
	return 0;
}

/**
 * load a source file from flash
 */
int32_t load_file(uint8_t file, char *buffer, uint32_t buffer_size)
{
	uint32_t file_id = PICOC_SOURCE_FILE_TYPE + file;
	uint8_t sector[PICOC_SECTOR_SIZE];
	int32_t  retval = 0;
	bool eof = false;

	for (uint32_t i = 0; i < buffer_size; i++) {
		// load sector
		if (!eof && (i % PICOC_SECTOR_SIZE == 0)) {
			retval = PIOS_FLASHFS_ObjLoad(pios_waypoints_settings_fs_id, file_id, i / PICOC_SECTOR_SIZE , (uint8_t *) &sector, PICOC_SECTOR_SIZE);
			eof = (retval != 0) ;
		}

		// copy sector content to buffer and check end of file
		buffer[i] = eof ? 0 : sector[i % PICOC_SECTOR_SIZE];
		eof |= (buffer[i] == 0);
	}
	return 0;
}

/**
 * save a source file to flash
 */
int32_t save_file(uint8_t file, char *buffer, uint32_t buffer_size)
{
	uint32_t file_id = PICOC_SOURCE_FILE_TYPE + file;
	uint8_t sector[PICOC_SECTOR_SIZE];
	int32_t retval = 0;
	bool eof = false;

	for (uint32_t i = 0; i < buffer_size; i++) {
		// copy sector content to buffer and check end of file
		sector[i % PICOC_SECTOR_SIZE] = eof ? 0 : buffer[i];
		eof |= (buffer[i] == 0);

		// save sector
		if ((i % PICOC_SECTOR_SIZE) == PICOC_SECTOR_SIZE - 1) {
			retval = PIOS_FLASHFS_ObjSave(pios_waypoints_settings_fs_id, file_id, i / PICOC_SECTOR_SIZE, (uint8_t *) &sector, PICOC_SECTOR_SIZE);
			if (eof) {
				break;
			}
		}
	}
	return retval;
}

/**
 * delete a source file
 */
int32_t delete_file(uint8_t file)
{
	uint32_t file_id = PICOC_SOURCE_FILE_TYPE + file;
	int32_t retval = PIOS_FLASHFS_ObjDelete(pios_waypoints_settings_fs_id, file_id, 0);
	return retval;
}

/**
 * format flash partition
 */
int32_t format_partition()
{
	int32_t retval = PIOS_FLASHFS_Format(pios_waypoints_settings_fs_id);
	return retval;
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */