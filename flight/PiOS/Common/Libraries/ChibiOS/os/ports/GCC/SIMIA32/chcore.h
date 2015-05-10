/*
    ChibiOS/RT - Copyright (C) 2006,2007,2008,2009,2010,
                 2011,2012,2013 Giovanni Di Sirio.

    This file is part of ChibiOS/RT.

    ChibiOS/RT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    ChibiOS/RT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

                                      ---

    A special exception to the GPL can be applied should you wish to distribute
    a combined work that includes ChibiOS/RT, without being obliged to provide
    the source code for any proprietary components. See the file exception.txt
    for full details of how and when the exception can be applied.
*/

/**
 * @addtogroup SIMIA32_CORE
 * @{
 */

#ifndef _CHCORE_H_
#define _CHCORE_H_

#include <ucontext.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*===========================================================================*/
/* Port constants.                                                           */
/*===========================================================================*/

/*===========================================================================*/
/* Port macros.                                                              */
/*===========================================================================*/

/*===========================================================================*/
/* Port configurable parameters.                                             */
/*===========================================================================*/

#if CH_DBG_ENABLE_STACK_CHECK
#error "option CH_DBG_ENABLE_STACK_CHECK not supported by this port"
#endif

#if !defined(_GNU_SOURCE)
#error "this port requires you to add -D_GNU_SOURCE to your CFLAGS"
#endif

/**
 * @brief   Stack size for the system idle thread.
 * @details This size depends on the idle thread implementation, usually
 *          the idle thread should take no more space than those reserved
 *          by @p PORT_INT_REQUIRED_STACK.
 */
#if !defined(PORT_IDLE_THREAD_STACK_SIZE) || defined(__DOXYGEN__)
#define PORT_IDLE_THREAD_STACK_SIZE     256
#endif

/**
 * @brief   Per-thread stack overhead for interrupts servicing.
 * @details This constant is used in the calculation of the correct working
 *          area size.
 *          This value can be zero on those architecture where there is a
 *          separate interrupt stack and the stack space between @p intctx and
 *          @p extctx is known to be zero.
 */
#if !defined(PORT_INT_REQUIRED_STACK) || defined(__DOXYGEN__)
#define PORT_INT_REQUIRED_STACK        32768 
#endif

/**
 * @brief   Typer time to be used to generate systick interrupt.
 * @details ITIMER_REAL
 *            Decrements in real time, and delivers SIGALRM upon expiration.
 *            This timer will fail when using debug breakpoints.
 *          ITIMER_VIRTUAL
 *            Decrements only when the process is executing, and delivers
 *            SIGVTALRM upon expiration.
 *          ITIMER_PROF
 *            Decrements both when the process executes and when the system
 *            is executing on behalf of the process. Delivers SIGPROF
 *            upon expiration.
 */
#if !defined(PORT_TIMER_TYPE) || defined(__DOXYGEN__)
#define PORT_TIMER_TYPE                 ITIMER_REAL
#endif

/**
 * @brief   Typer signal to be used to generate systick interrupt.
 * @details SIGALRM
 *            Will be delivered on expiration of ITIMER_REAL
 *          SIGVTALRM
 *            Will be delivered on expiration of ITIMER_VIRTUAL
 *          SIGPROF
 *            Will be delivered on expiration of ITIMER_PROF
 */
#if !defined(PORT_TIMER_SIGNAL) || defined(__DOXYGEN__)
#define PORT_TIMER_SIGNAL               SIGALRM
#endif

/*===========================================================================*/
/* Port derived parameters.                                                  */
/*===========================================================================*/
/*===========================================================================*/
/* Port exported info.                                                       */
/*===========================================================================*/

/**
 * Macro defining the a simulated architecture into x86.
 */
#define CH_ARCHITECTURE_SIMIA32

/**
 * Name of the implemented architecture.
 */
#define CH_ARCHITECTURE_NAME            "Simulator"

/**
 * @brief   Name of the architecture variant (optional).
 */
#define CH_CORE_VARIANT_NAME            "x86 (integer only)"

/**
 * @brief   Name of the compiler supported by this port.
 */
#define CH_COMPILER_NAME                "GCC " __VERSION__

/**
 * @brief   Port-specific information string.
 */
#if (TIMER_TYPE == ITIMER_REAL) || defined(__DOXYGEN__)
#define CH_PORT_INFO                    "Preemption through ITIMER_REAL"
#elif (TIMER_TYPE == ITIMER_VIRTUAL) || defined(__DOXYGEN__)
#define CH_PORT_INFO                    "Preemption through ITIMER_VIRTUAL"
#elif (TIMER_TYPE == ITIMER_PROF) || defined(__DOXYGEN__)
#define CH_PORT_INFO                    "Preemption through ITIMER_PROF"
#endif

/*===========================================================================*/
/* Port implementation part.                                                 */
/*===========================================================================*/

/**
 * @brief   Base type for stack and memory alignment.
 */
typedef struct {
  uint8_t a[16];
} stkalign_t __attribute__((aligned(16)));

/**
 * @brief   Interrupt saved context.
 * @details This structure represents the stack frame saved during a
 *          preemption-capable interrupt handler.
 */
struct extctx {
};

/**
 * @brief   System saved context.
 * @details This structure represents the inner stack frame during a context
 *          switching.
 */
struct intctx {
  ucontext_t uc;
};

/**
 * @brief   Platform dependent part of the @p Thread structure.
 * @details This structure usually contains just the saved stack pointer
 *          defined as a pointer to a @p intctx structure.
 */
struct context {
  ucontext_t uc;
};

/**
 * @brief   Platform dependent part of the @p chThdCreateI() API.
 * @details This code usually setup the context switching frame represented
 *          by an @p intctx structure.
 */
#define SETUP_CONTEXT(workspace, wsize, pf, arg) {                      \
  if (getcontext(&tp->p_ctx.uc) < 0)                                    \
    port_halt();                                                        \
  tp->p_ctx.uc.uc_stack.ss_sp = workspace;                              \
  tp->p_ctx.uc.uc_stack.ss_size = wsize;                                \
  tp->p_ctx.uc.uc_stack.ss_flags = 0;                                   \
  tp->p_ctx.uc.uc_link = NULL;						\
  makecontext(&tp->p_ctx.uc, (void(*)(void))_port_thread_start, 2, pf, (void *)arg);     \
}

/**
 * @brief   Enforces a correct alignment for a stack area size value.
 */
#define STACK_ALIGN(n) ((((n) - 1) | (sizeof(stkalign_t) - 1)) + 1)

/**
 * @brief   Computes the thread working area global size.
 */
#define THD_WA_SIZE(n) STACK_ALIGN(sizeof(Thread) +                     \
                                   sizeof(void *) * 4 +                 \
                                   sizeof(struct intctx) +              \
                                   sizeof(struct extctx) +              \
                                   (n) + (PORT_INT_REQUIRED_STACK))

/**
 * @brief   Static working area allocation.
 * @details This macro is used to allocate a static thread working area
 *          aligned as both position and size.
 */
#define WORKING_AREA(s, n) stkalign_t s[THD_WA_SIZE(n) / sizeof(stkalign_t)]

/**
 * @brief   IRQ prologue code.
 * @details This macro must be inserted at the start of all IRQ handlers
 *          enabled to invoke system APIs.
 */
#define PORT_IRQ_PROLOGUE()

/**
 * @brief   IRQ epilogue code.
 * @details This macro must be inserted at the end of all IRQ handlers
 *          enabled to invoke system APIs.
 */
#define PORT_IRQ_EPILOGUE()

/**
 * @brief   IRQ handler function declaration.
 * @note    @p id can be a function name or a vector number depending on the
 *          port implementation.
 */
#define PORT_IRQ_HANDLER(id) void id(int sig)

#ifdef __cplusplus
extern "C" {
#endif
  void port_init(void);
  void port_lock(void);
  void port_unlock(void);
  void port_lock_from_isr(void);
  void port_unlock_from_isr(void);
  void port_disable(void);
  void port_suspend(void);
  void port_enable(void);
  void port_wait_for_interrupt(void);
  void port_halt(void);
  void port_switch(Thread *ntp, Thread *otp);

  void _port_thread_start(void (*func)(int), int arg);
#ifdef __cplusplus
}
#endif

#endif /* _CHCORE_H_ */

/** @} */
