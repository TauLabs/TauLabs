/**
  ******************************************************************************
  * @addtogroup PIOS PIOS Core hardware abstraction layer
  * @{
  * @addtogroup PIOS_RESET Reset functions
  * @brief Hardware functions to deal with the reset register
  * @{
  *
  * @file       pios_reset.h
  * @author     Kenn Sebesta, Copyright (C) 2015.
  * @brief      Reset information
  *
  ******************************************************************************/

#ifndef PIOS_RESET_PRIV_H
#define PIOS_RESET_PRIV_H

// Private types
enum reboot_flags {
    PIOS_RESET_FLAG_UNDEFINED,            // Undefined reason
    PIOS_RESET_FLAG_BROWNOUT,             // POR/PDR or BOR reset
    PIOS_RESET_FLAG_PIN,                  // Pin reset
    PIOS_RESET_FLAG_POWERON,              // POR/PDR reset
    PIOS_RESET_FLAG_SOFTWARE,             // Software reset
    PIOS_RESET_FLAG_INDEPENDENT_WATCHDOG, // Independent Watchdog reset
    PIOS_RESET_FLAG_WINDOW_WATCHDOG,      // Window Watchdog reset
    PIOS_RESET_FLAG_LOW_POWER,            // Low Power reset
};


void PIOS_RESET_Clear (void);
int16_t PIOS_RESET_GetResetReason (void);

#endif /* PIOS_RESET_PRIV_H */
