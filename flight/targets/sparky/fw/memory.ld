__main_stack_size__     = 0x0400;
__process_stack_size__  = 0x0400;

MEMORY
{
    bd_info (r) : org = 0x08004000 - 0x80, len = 0x000080
    flash : org = 0x08000000 + 48k, len = 256k - 48k
    ram : org = 0x20000000, len = 40k
    ccmram : org = 0x10000000, len = 8k
}
