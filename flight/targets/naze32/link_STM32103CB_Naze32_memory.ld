MEMORY
{
    FLASH         : ORIGIN = 0x08000000,        LENGTH = 128K - 100 - 12K
    /* FW_INFO should be filled when the tlfw is generated */
    FW_INFO       : ORIGIN = 0x0801D000 - 100,  LENGTH = 100
    /* Putting anything in EEPROM will overwrite saved settings */
    EEPROM        : ORIGIN = 0x0801D000,        LENGTH = 12K
    SRAM          : ORIGIN = 0x20000000,        LENGTH = 20K
}
