
import socket
import taulabs
import time
import array

class Telemetry():
	"""
	Provides a basic telemetry connection to a flight controller
	"""

	def __init__(self, uavtalk):
		self.uavtalk_parser = uavtalk
		self.sock = 0

		# handy copy
		self.uavo_defs = uavtalk.uavo_defs
		self.gcs_telemetry = {v: k for k, v in self.uavo_defs.items() if v.meta['name']=="GCSTelemetryStats"}.items()[0][0]

		self.uavo_list = taulabs.uavo_list.UAVOList(self.uavo_defs)

	def open_network(self):
		""" Open a socket on localhost port 9000 """

		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect(("127.0.0.1", 9000))
		self.sock = s

	def close_network(self):
		""" Close network socket """

		s.sock.shutdown()
		s.sock.close()

	def serviceConnection(self):
		"""
		Receive and parse data from a network connection and handle the basic
		handshaking with the flight controller
		"""

		updated_objects = 0

		for c in self.__receive():
			self.uavtalk_parser.processByte(ord(c))

			if self.uavtalk_parser.state == taulabs.uavtalk.UavTalk.STATE_COMPLETE:
				obj  = self.uavtalk_parser.getLastReceivedObject(timestamp=round(time.time() * 1000))

				if obj is not None:

					# TODO: insert code to handle interest updates here. Default behavior is simply
					# to store them all
					updated_objects = updated_objects + 1
					self.uavo_list.append(obj)

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

		return updated_objects, self.uavo_list

	def __send(self, msg):
		""" Send a string out the TCP socket """

		totalsent = 0
		while totalsent < len(msg):
			sent = self.sock.send(msg[totalsent:])
			if sent == 0:
				raise RuntimeError("socket connection broken")
			totalsent = totalsent + sent

	def __receive(self):
		""" Fetch available data from TCP socket """

		MSGLEN = 32

		msg = ''
		while len(msg) < MSGLEN:
			chunk = self.sock.recv(MSGLEN-len(msg))
			if chunk == '':
				raise RuntimeError("socket connection broken")
			msg = msg + chunk
		return msg


