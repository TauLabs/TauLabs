MEMORY
{
    BD_INFO (r)   : ORIGIN = 0x08004000 - 0x80, LENGTH =            0x000080
    FLASH (rx)    : ORIGIN = 0x08004000,        LENGTH = 0x03C000
    SRAM (rwx)    : ORIGIN = 0x20000000,        LENGTH = 0x00A000
    CCSRAM (rw)   : ORIGIN = 0x10000000,        LENGTH = 0x002000
}
