/**
 ******************************************************************************
 *
 * @file       pios_tcp.c   
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      TCP commands. Inits UDPs, controls UDPs & Interupt handlers.
 * @see        The GNU Public License (GPL) Version 3
 * @defgroup   PIOS_UDP UDP Functions
 * @{
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


/* Project Includes */
#include "pios.h"

#if defined(PIOS_INCLUDE_TCP)

#include <pios_tcp_priv.h>
#include "pios_thread.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>

/* We need a list of TCP devices */

#define PIOS_TCP_MAX_DEV 16
static int8_t pios_tcp_num_devices = 0;

static pios_tcp_dev pios_tcp_devices[PIOS_TCP_MAX_DEV];



/* Provide a COM driver */
static void PIOS_TCP_ChangeBaud(uintptr_t udp_id, uint32_t baud);
static void PIOS_TCP_RegisterRxCallback(uintptr_t udp_id, pios_com_callback rx_in_cb, uintptr_t context);
static void PIOS_TCP_RegisterTxCallback(uintptr_t udp_id, pios_com_callback tx_out_cb, uintptr_t context);
static void PIOS_TCP_TxStart(uintptr_t udp_id, uint16_t tx_bytes_avail);
static void PIOS_TCP_RxStart(uintptr_t udp_id, uint16_t rx_bytes_avail);

const struct pios_com_driver pios_tcp_com_driver = {
	.set_baud   = PIOS_TCP_ChangeBaud,
	.tx_start   = PIOS_TCP_TxStart,
	.rx_start   = PIOS_TCP_RxStart,
	.bind_tx_cb = PIOS_TCP_RegisterTxCallback,
	.bind_rx_cb = PIOS_TCP_RegisterRxCallback,
};


static pios_tcp_dev * find_tcp_dev_by_id(uint8_t tcp)
{
	if (tcp >= pios_tcp_num_devices) {
		/* Undefined UDP port for this board (see pios_board.c) */
		PIOS_Assert(0);
		return NULL;
	}
	
	/* Get a handle for the device configuration */
	return &(pios_tcp_devices[tcp]);
}

/**
 * RxTask
 */
static void PIOS_TCP_RxTask(void *tcp_dev_n)
{
	pios_tcp_dev *tcp_dev = (pios_tcp_dev*)tcp_dev_n;
	
	const int INCOMING_BUFFER_SIZE = 16;
	char incoming_buffer[INCOMING_BUFFER_SIZE];
	int error;
	/**
	 * com devices never get closed except by application "reboot"
	 * we also never give up our mutex except for waiting
	 */
	while (1) {
	
		do
		{
			/* Polling the fd has to be executed in thread suspended mode
			 * to get a correct errno value. */
			PIOS_Thread_Scheduler_Suspend();

			tcp_dev->socket_connection = accept(tcp_dev->socket, NULL, NULL);
			error = errno;

			PIOS_Thread_Scheduler_Resume();

			PIOS_Thread_Sleep(1);
		} while (tcp_dev->socket_connection == -1 && (error == EINTR || error == EAGAIN));

		if (tcp_dev->socket_connection < 0) {
			int error = errno;
			(void)error;
			perror("Accept failed");
			close(tcp_dev->socket);
			exit(EXIT_FAILURE);
		}
		
		/* Set socket nonblocking. */
	    int flags;
	    if ((flags = fcntl(tcp_dev->socket_connection, F_GETFL, 0)) == -1) {
	    }
	    if (fcntl(tcp_dev->socket_connection, F_SETFL, flags | O_NONBLOCK) == -1) {
	    }


		fprintf(stderr, "Connection accepted\n");

		while (1) {
			// Received is used to track the scoket whereas the dev variable is only updated when it can be

			/* Polling the fd has to be executed in thread suspended mode
			 * to get a correct errno value. */
			PIOS_Thread_Scheduler_Suspend();

			int result = read(tcp_dev->socket_connection, incoming_buffer, INCOMING_BUFFER_SIZE);
			error = errno;

			PIOS_Thread_Scheduler_Resume();
			
			if (result > 0 && tcp_dev->rx_in_cb) {

				bool rx_need_yield = false;

				tcp_dev->rx_in_cb(tcp_dev->rx_in_context, (uint8_t*)incoming_buffer, result, NULL, &rx_need_yield);

	#if defined(PIOS_INCLUDE_FREERTOS)
				// Not sure about this
				if (rx_need_yield) {
					taskYIELD();
				}
	#endif	/* PIOS_INCLUDE_FREERTOS */

			}

			if (result == 0) {
				break;
			}

			if (result == -1) {
				if (error == EAGAIN)
					PIOS_Thread_Sleep(1);
				else if (error == EINTR)
					(void)error;
				else
					break;
			}
		}
		
		if (shutdown(tcp_dev->socket_connection, SHUT_RDWR) == -1) {
			//perror("can not shutdown socket");
			//close(tcp_dev->socket_connection);
			//exit(EXIT_FAILURE);
		}
		close(tcp_dev->socket_connection);
		tcp_dev->socket_connection = 0;
	}
}


/**
 * Open TCP socket
 */
struct pios_thread *tcpRxTaskHandle;
int32_t PIOS_TCP_Init(uintptr_t *tcp_id, const struct pios_tcp_cfg * cfg)
{
	
	pios_tcp_dev *tcp_dev = &pios_tcp_devices[pios_tcp_num_devices];
	
	pios_tcp_num_devices++;
	
	/* initialize */
	tcp_dev->rx_in_cb = NULL;
	tcp_dev->tx_out_cb = NULL;
	tcp_dev->cfg=cfg;
	
	/* assign socket */
	tcp_dev->socket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

	int optval=1;

        /* Allow reuse of address if you restart. */
        setsockopt(tcp_dev->socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

        /* Also request low-latency (don't store up data to conserve packets */
        setsockopt(tcp_dev->socket, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));

	memset(&tcp_dev->server, 0, sizeof(tcp_dev->server));
	memset(&tcp_dev->client, 0, sizeof(tcp_dev->client));

	tcp_dev->server.sin_family = AF_INET;
	tcp_dev->server.sin_addr.s_addr = INADDR_ANY; //inet_addr(tcp_dev->cfg->ip);
	tcp_dev->server.sin_port = htons(tcp_dev->cfg->port);

	/* set socket options */
    int value = 1;
    if (setsockopt(tcp_dev->socket, SOL_SOCKET, SO_REUSEADDR, (char*)&value, sizeof(value)) == -1) {

    }

	int res= bind(tcp_dev->socket, (struct sockaddr*)&tcp_dev->server, sizeof(tcp_dev->server));
	if (res == -1) {
		perror("Binding socket failed\n");
		exit(EXIT_FAILURE);
	}
	
	res = listen(tcp_dev->socket, 10);
	if (res == -1) {
		perror("Socket listen failed\n");
		exit(EXIT_FAILURE);
	}
	
	/* Set socket nonblocking. */
    int flags;
    if ((flags = fcntl(tcp_dev->socket, F_GETFL, 0)) == -1) {
    }
    if (fcntl(tcp_dev->socket, F_SETFL, flags | O_NONBLOCK) == -1) {
    }

	tcpRxTaskHandle = PIOS_Thread_Create(
			PIOS_TCP_RxTask, "pios_tcp_rx", PIOS_THREAD_STACK_SIZE_MIN, tcp_dev, PIOS_THREAD_PRIO_HIGHEST);
	
	printf("tcp dev %i - socket %i opened - result %i\n", pios_tcp_num_devices - 1, tcp_dev->socket, res);
	
	*tcp_id = pios_tcp_num_devices - 1;
	
	return res;
}


void PIOS_TCP_ChangeBaud(uintptr_t tcp_id, uint32_t baud)
{
	/**
	 * doesn't apply!
	 */
}


static void PIOS_TCP_RxStart(uintptr_t tp_id, uint16_t rx_bytes_avail)
{
	/**
	 * lazy!
	 */
}


static void PIOS_TCP_TxStart(uintptr_t tcp_id, uint16_t tx_bytes_avail)
{
	pios_tcp_dev *tcp_dev = find_tcp_dev_by_id(tcp_id);
	
	PIOS_Assert(tcp_dev);
	
	int32_t length,rem;
	
	/**
	 * we send everything directly whenever notified of data to send (lazy!)
	 */
	if (tcp_dev->tx_out_cb) {
		while (tx_bytes_avail > 0) {
			bool tx_need_yield = false;
			length = (tcp_dev->tx_out_cb)(tcp_dev->tx_out_context, tcp_dev->tx_buffer, PIOS_TCP_RX_BUFFER_SIZE, NULL, &tx_need_yield);
			rem = length;
			while (rem > 0) {
				ssize_t len = 0;
				if (tcp_dev->socket_connection != 0) {
					len = write(tcp_dev->socket_connection, tcp_dev->tx_buffer, length);
				}
				if (len <= 0) {
					rem = 0;
				} else {
					rem -= len;
				}
			}
			tx_bytes_avail -= length;
#if defined(PIOS_INCLUDE_FREERTOS)
			// Not sure about this
			if (tx_need_yield) {
				taskYIELD();
			}
#endif	/* PIOS_INCLUDE_FREERTOS */
		}
	}
}

static void PIOS_TCP_RegisterRxCallback(uintptr_t tcp_id, pios_com_callback rx_in_cb, uintptr_t context)
{
	pios_tcp_dev *tcp_dev = find_tcp_dev_by_id(tcp_id);
	
	PIOS_Assert(tcp_dev);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	tcp_dev->rx_in_context = context;
	tcp_dev->rx_in_cb = rx_in_cb;
}

static void PIOS_TCP_RegisterTxCallback(uintptr_t tcp_id, pios_com_callback tx_out_cb, uintptr_t context)
{
	pios_tcp_dev *tcp_dev = find_tcp_dev_by_id(tcp_id);
	
	PIOS_Assert(tcp_dev);
	
	/* 
	 * Order is important in these assignments since ISR uses _cb
	 * field to determine if it's ok to dereference _cb and _context
	 */
	tcp_dev->tx_out_context = context;
	tcp_dev->tx_out_cb = tx_out_cb;
}

#endif
