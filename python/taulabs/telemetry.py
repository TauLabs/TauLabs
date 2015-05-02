import socket
import taulabs
import time
import array
import select
import errno

class Telemetry():
	"""
	Provides a basic telemetry connection to a flight controller
	"""

	def __init__(self, uavtalk):
		self.uavtalk_parser = uavtalk
		self.sock = None

		# handy copy
		self.uavo_defs = uavtalk.uavo_defs
		self.gcs_telemetry = {v: k for k, v in self.uavo_defs.items() if v.meta['name']=="GCSTelemetryStats"}.items()[0][0]

		self.recv_buf = ''

		self.uavo_list = taulabs.uavo_list.UAVOList(self.uavo_defs)

	def open_network(self, host="127.0.0.1", port=9000):
		""" Open a socket on localhost port 9000 """

		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((host, port))

		s.setblocking(0)

		self.sock = s

	def close_network(self):
		""" Close network socket """

		self.sock.close()
		self.sock=None

	def __handleFrame(self, frame):
		updated=0

		for c in frame:
			self.uavtalk_parser.processByte(ord(c))

			if self.uavtalk_parser.state == taulabs.uavtalk.UavTalk.STATE_COMPLETE:
				obj  = self.uavtalk_parser.getLastReceivedObjectInstance(timestamp=round(time.time() * 1000))

				if obj is not None:
					updated += 1

					# TODO: insert code to handle interest updates here. Default behavior is simply
					# to store them all
					self.uavo_list.append(obj)

					print obj

					if obj.name == "FlightTelemetryStats":
						# Handle the telemetry hanshaking

						(DISCONNECTED, HANDSHAKE_REQ, HANDSHAKE_ACK, CONNECTED) = (0,1,2,3)

						if obj.Status == DISCONNECTED:
							print "Disconnected"
							# Request handshake
							send_obj = self.gcs_telemetry.tuple_class._make(["GCSTelemetryStats", round(time.time() * 1000), 
								self.gcs_telemetry.id, 0, 0, 0, 0, 0, HANDSHAKE_REQ])
						elif obj.Status == HANDSHAKE_ACK:
							print "Handshake ackd"
							# Say connected
							send_obj = self.gcs_telemetry.tuple_class._make(["GCSTelemetryStats", round(time.time() * 1000), 
								self.gcs_telemetry.id, 0, 0, 0, 0, 0, CONNECTED])
						elif obj.Status == CONNECTED:
							print "Connected"
							send_obj = self.gcs_telemetry.tuple_class._make(["GCSTelemetryStats", round(time.time() * 1000), 
								self.gcs_telemetry.id, 0, 0, 0, 0, 0, CONNECTED])
						packet = self.uavtalk_parser.sendSingleObject(send_obj)
						self.__send(packet)

		return updated


	def serviceConnection(self, timeout=None):
		"""
		Receive and parse data from a network connection and handle the basic
		handshaking with the flight controller
		"""

		if timeout is not None:
			finishTime = time.time()+timeout
		else:
			finishTime = None

		updated_objects = 0

		for i in self.__receive(finishTime):
			updated_objects += self.__handleFrame(i)

		return updated_objects, self.uavo_list

	def __send(self, msg):
		""" Send a string out the TCP socket """

		totalsent = 0
		while totalsent < len(msg):
			sent = self.sock.send(msg[totalsent:])
			if sent == 0:
				raise RuntimeError("socket connection broken")
			totalsent = totalsent + sent

	def __receive(self, finishTime):
		""" Fetch available data from TCP socket """

		MSGLEN = 32
		first=True

		while len(self.recv_buf) < MSGLEN:
			now = time.time()
			if finishTime is None: 
				select.select([self.sock], [], [])
			elif now < finishTime:
				select.select([self.sock], [], [], finishTime-now)
			else:
				if not first:
					return None

			first=False

			try:
				chunk = self.sock.recv(1024)
				if chunk == '':
					raise RuntimeError("socket connection broken")

				self.recv_buf=self.recv_buf + chunk
			except socket.timeout:
				pass
			except socket.error,e:
				if e.errno != errno.EAGAIN:
					raise


		ret=self.recv_buf[0:MSGLEN]
		self.recv_buf=self.recv_buf[MSGLEN:]

		return ret

