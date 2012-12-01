/*
 * syscalls.c
 *
 *  Created on: 03.12.2009
 *      Author: Martin Thomas, 3BSD license
 */

//#define SBRK_VERBOSE 1


#include <reent.h>
#include <errno.h>
#include <stdlib.h> /* abort */
#include <sys/types.h>
#include <sys/stat.h>

#include "dcc_stdio.h"

#undef errno
extern int errno;

int _kill(int pid, int sig)
{
    errno = EINVAL;
    return -1;
}

void _exit(int status)
{
    dbg_write_str("_exit called with parameter\n");
    while (1)
    {
    }
}

int _getpid(void)
{
    return 1;
}

extern char _end; /* Defined by the linker */
static char *heap_end;

char* get_heap_end(void)
{
    return (char*) heap_end;
}

char* get_stack_top(void)
{
    //return (char*) __get_MSP();
    //return (char*) __get_PSP();
}

caddr_t _sbrk(int incr)
{
    char *prev_heap_end;
#if SBRK_VERBOSE
    dbg_write_str("_sbrk called with incr \n");
#endif
    if (heap_end == 0)
    {
        heap_end = &_end;
    }
    prev_heap_end = heap_end;
    if (heap_end + incr > get_stack_top())
    {
        dbg_write_str("Heap and stack collision\n");
        abort();
    }
    return (caddr_t) prev_heap_end;
}

int _close(int file)
{
    return -1;
}

int _fstat(int file, struct stat *st)
{
    return 0;
}

int _isatty(int file)
{
    return 1;
}

int _lseek(int file, int ptr, int dir)
{
    return 0;
}

int _read(int file, char *ptr, int len)
{
    return 0;
}

int _write(int file, char *ptr, int len)
{
    return len;
}
