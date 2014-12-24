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

#include "ch.h"
#include "hal.h"

#include <ucontext.h>

/*===========================================================================*/
/* Port interrupt handlers.                                                  */
/*===========================================================================*/

CH_IRQ_HANDLER(port_tick_signal_handler) {
  CH_IRQ_PROLOGUE();

  chSysLockFromIsr();
  chSysTimerHandlerI();
  chSysUnlockFromIsr();

  CH_IRQ_EPILOGUE();

  dbg_check_lock();
  if (chSchIsPreemptionRequired())
    chSchDoReschedule();
  dbg_check_unlock();
}

/*===========================================================================*/
/* Port local functions.                                                     */
/*===========================================================================*/

static void _setcontext(void *arg) {
  Thread *ntp = (Thread*)arg;
  if (setcontext(&ntp->p_ctx.uc) < 0)
    port_halt();
}

/*===========================================================================*/
/* Port exported functions.                                                  */
/*===========================================================================*/

/**
 * @brief   Port-related initialization code.
 * @note    This function is usually empty.
 */
void port_init(void) {
}

/**
 * @brief   Kernel-lock action.
 * @details Usually this function just disables interrupts but may perform more
 *          actions.
 */
void port_lock(void) {
  sigset_t set;
  if (sigfillset(&set) < 0)
    port_halt();
  if (sigprocmask(SIG_BLOCK, &set, NULL) > 0)
    port_halt();
}

/**
 * @brief   Kernel-unlock action.
 * @details Usually this function just enables interrupts but may perform more
 *          actions.
 */
void port_unlock(void) {
  sigset_t set;
  if (sigfillset(&set) < 0)
    port_halt();
  if (sigprocmask(SIG_UNBLOCK, &set, NULL) > 0)
    port_halt();
}

/**
 * @brief   Kernel-lock action from an interrupt handler.
 * @details This function is invoked before invoking I-class APIs from
 *          interrupt handlers. The implementation is architecture dependent,
 *          in its simplest form it is void.
 */
void port_lock_from_isr(void) {
    sigset_t set;
    if (sigfillset(&set) < 0)
      port_halt();
    /* The timer signal is masked on entry of systick automatically. */
    if (sigdelset(&set, PORT_TIMER_SIGNAL))
      port_halt();
    if (sigprocmask(SIG_BLOCK, &set, NULL) > 0)
      port_halt();
}

/**
 * @brief   Kernel-unlock action from an interrupt handler.
 * @details This function is invoked after invoking I-class APIs from interrupt
 *          handlers. The implementation is architecture dependent, in its
 *          simplest form it is void.
 */
void port_unlock_from_isr(void) {
    sigset_t set;
    if (sigfillset(&set) < 0)
      port_halt();
    /* The timer signal is unmasked on exit of systick automatically. */
    if (sigdelset(&set, PORT_TIMER_SIGNAL))
      port_halt();
    if (sigprocmask(SIG_UNBLOCK, &set, NULL) > 0)
      port_halt();
}

/**
 * @brief   Disables all the interrupt sources.
 * @note    Of course non-maskable interrupt sources are not included.
 */
void port_disable(void) {
}

/**
 * @brief   Disables the interrupt sources below kernel-level priority.
 * @note    Interrupt sources above kernel level remains enabled.
 */
void port_suspend(void) {
}

/**
 * @brief   Enables all the interrupt sources.
 */
void port_enable(void) {
}

/**
 * @brief   Enters an architecture-dependent IRQ-waiting mode.
 * @details The function is meant to return when an interrupt becomes pending.
 *          The simplest implementation is an empty function or macro but this
 *          would not take advantage of architecture-specific power saving
 *          modes.
 */
void port_wait_for_interrupt(void) {
}

/**
 * @brief   Halts the system.
 * @details This function is invoked by the operating system when an
 *          unrecoverable error is detected (for example because a programming
 *          error in the application code that triggers an assertion while in
 *          debug mode).
 */
void port_halt(void) {
  port_disable();
  exit(2);
}

/**
 * @brief   Performs a context switch between two threads.
 * @details This is the most critical code in any port, this function
 *          is responsible for the context switch between 2 threads.
 * @note    The implementation of this code affects <b>directly</b> the context
 *          switch performance so optimize here as much as you can.
 *
 * @param[in] ntp       the thread to be switched in
 * @param[in] otp       the thread to be switched out
 */
void port_switch(Thread *ntp, Thread *otp) {
  /* Create temporary context to perform swap. */
  static uint8_t tempstack[PORT_INT_REQUIRED_STACK];
  static ucontext_t tempctx;
  if (getcontext(&tempctx) < 0)
    port_halt();
  tempctx.uc_stack.ss_sp = tempstack;
  tempctx.uc_stack.ss_size = sizeof(tempstack);
  tempctx.uc_stack.ss_flags = 0;
  makecontext(&tempctx, (void(*)(void))_setcontext, 1, ntp);

  /* Save running thread, jump to temporary context. */
  if (swapcontext(&otp->p_ctx.uc, &tempctx) < 0)
    port_halt();
}

/**
 * @brief   Start a thread by invoking its work function.
 * @details If the work function returns @p chThdExit() is automatically
 *          invoked.
 */
void _port_thread_start(void) {
  asm volatile ("push %ecx                                      \n\t"
                "push %edx");
  chSysUnlock();
  asm volatile ("pop %edx                                       \n\t"
                "pop %ecx                                       \n\t"
                "push %edx                                      \n\t"
                "call *%ecx                                     \n\t"
                "push %eax                                      \n\t"
                "call chThdExit");
}

/** @} */
