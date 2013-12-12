#ifndef __LINUX_STRINGIFY_H
#define __LINUX_STRINGIFY_H

/* Indirect stringification.  Doing two levels allows the parameter to be a
 * macro itself.  For example, compile with -DFOO=bar, __stringify(FOO)
 * converts to "bar".
 *
 * Copied from linux kernel:
 *    http://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/include/linux/stringify.h
 */
#define __stringify_1(x...)	#x
#define __stringify(x...)	__stringify_1(x)

#endif	/* !__LINUX_STRINGIFY_H */
