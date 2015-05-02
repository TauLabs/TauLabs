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
		self.send_buf = ''

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

		self.send_buf += msg

		self.__select(0)

	# Call select and do one set of IO operations.
	def __select(self, finishTime):
		rdSet=[]
		wrSet=[]

		didStuff=False

		if len(self.recv_buf) < 1024:
			rdSet.append(self.sock)

		if len(self.send_buf) > 0:
			wrSet.append(self.sock)

		now = time.time()
		if finishTime is None: 
			r,w,e = select.select(rdSet, wrSet, [])
		else:
			tm = finishTime-now
			if tm < 0: tm=0

			r,w,e = select.select(rdSet, wrSet, [], tm)

		if r:
			# Shouldn't throw an exception-- they just told us
			# it was ready for read.
			chunk = self.sock.recv(1024)
			if chunk == '':
				raise RuntimeError("socket closed")

			self.recv_buf=self.recv_buf + chunk

			didStuff=True

		if w:
			written = self.sock.send(self.send_buf)
			self.send_buf = self.send_buf[written:]

			didStuff=True

		return didStuff


	def __receive(self, finishTime):
		""" Fetch available data from TCP socket """

		MSGLEN = 32
		first=True

		# Always do some minimal IO if possible
		self.__select(0)

		while (len(self.recv_buf) < MSGLEN) and self.__select(finishTime):
			pass

		if len(self.recv_buf) < MSGLEN:
			return None

		ret=self.recv_buf[0:MSGLEN]
		self.recv_buf=self.recv_buf[MSGLEN:]

		return ret

