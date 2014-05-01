
class UavTalk():
	"""
	Parses and generates a UAVTalk stream. This is the class is used for both log parsing
	and also maintaing a telemetry stream. The set of objects that can be parsed are
	determined by the uavo_defs field. 
	"""

	# Constants used for UAVTalk parsing
	(STATE_SYNC,STATE_TYPE,STATE_SIZE,STATE_OBJID,STATE_INSTID,STATE_TIMESTAMP,STATE_DATA,STATE_CS,STATE_COMPLETE,STATE_ERROR) = (0,1,2,3,4,5,6,7,8,9)
	(MIN_HEADER_LENGTH, MAX_HEADER_LENGTH, MAX_PAYLOAD_LENGTH) = (8, 12, (256-12))
	(SYNC_VAL) = (0x3C)
	(TYPE_MASK, TYPE_VER) = (0x78, 0x20)
	(TIMESTAMPED) = (0x80)
	(TYPE_OBJ, TYPE_OBJ_REQ, TYPE_OBJ_ACK, TYPE_ACK, TYPE_NACK, TYPE_OBJ_TS, TYPE_OBJ_ACK_TS) = (0x00, 0x01, 0x02, 0x03, 0x04, 0x80, 0x82)

	def __init__(self, uavo_defs):
		self.uavo_defs = uavo_defs # The set of UAVO types to parse

		self.state = UavTalk.STATE_COMPLETE
		self.packetSize = 0
		self.rxPacketLength = 0
		self.rxCount = 0  # counts the number of bytes to receive in each state
		self.type = 0
		self.length = 0
		self.instanceLength = 0
		self.timestampLength = 0
		self.timestamp = 0
		self.rxBuffer = "" # needs to be some kind of byte array

		# These are used for accounting for timestamp wraparound
		self.timestampBase = 0
		self.lastTimestamp = 0

		self.obj = None # stores the object type when found
	
	def getLastReceivedObject(self, timestamp=0):
		"""
		Return the object that was last updated
		"""

		if self.state == UavTalk.STATE_COMPLETE and self.obj is not None:
			if self.type == UavTalk.TYPE_OBJ_TS:
				return ('{0:08x}'.format(self.objId), self.rxBuffer, self.timestamp)
			else:
				return ('{0:08x}'.format(self.objId), self.rxBuffer, timestamp)

	def processByte(self, rxbyte):
		"""
		Process a byte from a telemetry stream. This implements a simple state machine
		to know which part of a UAVTalk packet this byte is and respond to it appropriately.
		The result of this parsing can be accessed with getLastReceivedObject
		"""

		if self.state == UavTalk.STATE_ERROR or self.state == UavTalk.STATE_COMPLETE:
			self.state = UavTalk.STATE_SYNC
	
		if self.rxPacketLength < 0xffff:
			self.rxPacketLength = self.rxPacketLength + 1   # update packet byte count
	
		# Receive state machine
		if self.state == UavTalk.STATE_SYNC:
			if rxbyte != UavTalk.SYNC_VAL:
				return
			
			# Initialize and update the CRC, then advance state machine
			self.cs = 0
			self.cs = self.__updateCRC(rxbyte)
			self.rxPacketLength = 1
			self.state = UavTalk.STATE_TYPE

		elif self.state == UavTalk.STATE_TYPE:
			
			# update the CRC
			self.cs = self.__updateCRC(rxbyte)
			
			if (rxbyte & UavTalk.TYPE_MASK) != UavTalk.TYPE_VER:
				self.state = UavTalk.STATE_ERROR
				return
			
			# Store the type and advance the state machine
			self.type = rxbyte & ~UavTalk.TYPE_MASK
			self.packetSize = 0
			self.state = UavTalk.STATE_SIZE
			self.rxCount = 0

		elif self.state == UavTalk.STATE_SIZE:
			
			# update the CRC
			self.cs = self.__updateCRC(rxbyte)
			
			if self.rxCount == 0:
				# parse the first byte of the size
				self.packetSize = self.packetSize + rxbyte
				self.rxCount = self.rxCount + 1
				return

			self.packetSize = self.packetSize + (rxbyte << 8)
			
			
			if self.packetSize < UavTalk.MIN_HEADER_LENGTH or self.packetSize > UavTalk.MAX_HEADER_LENGTH + UavTalk.MAX_PAYLOAD_LENGTH:
				self.state = UavTalk.STATE_ERROR
				return
			
			self.rxCount = 0
			self.objId = 0
			self.state = UavTalk.STATE_OBJID

		elif self.state == UavTalk.STATE_OBJID:
			
			# update the CRC
			self.cs = self.__updateCRC(rxbyte)
			
			self.objId = self.objId + (rxbyte << (8*self.rxCount))
			self.rxCount = self.rxCount + 1

			if self.rxCount < 4:
				# get all of the bytes in the ID
				return
			
			# Search for object.
			uavo_key = '{0:08x}'.format(self.objId)
			if not uavo_key in self.uavo_defs:
				print "Unknown object 0x" + uavo_key
				self.obj = None
			else:
				self.obj = self.uavo_defs[uavo_key]
				
			# Determine data length
			if self.type == UavTalk.TYPE_OBJ_REQ or self.type == UavTalk.TYPE_ACK or self.type == UavTalk.TYPE_NACK:
				self.length = 0
				self.instanceLength = 0
			else:
				if self.obj is not None:
					self.instanceLength =  0 if self.obj.meta['is_single_inst'] else 2
					self.timestampLength = 2 if self.type == UavTalk.TYPE_OBJ_TS or self.type == UavTalk.TYPE_OBJ_ACK_TS else 0
					self.length = self.obj.get_size_of_data()

				else:
					# We don't know if it's a multi-instance object, so just assume it's 0.
					self.instanceLength = 0
					self.length = self.packetSize - self.rxPacketLength

			# Check length and determine next state
			if self.length >= UavTalk.MAX_PAYLOAD_LENGTH:
				self.state = UavTalk.STATE_ERROR
				return
			
			# Check the lengths match
			if (self.rxPacketLength + self.instanceLength + self.timestampLength + self.length) != self.packetSize:
				# packet error - mismatched packet size
				self.state = UavTalk.STATE_ERROR
				return
			
			self.instId = 0
			if self.type == UavTalk.TYPE_NACK:
				# If this is a NACK, we skip to Checksum
				self.state = UavTalk.STATE_CS
			elif self.obj is not None and not self.obj.meta['is_single_inst']:
				# Check if this is a single instance object (i.e. if the instance ID field is coming next)
				self.state = UavTalk.STATE_INSTID
			elif self.obj is not None and self.type & UavTalk.TIMESTAMPED:
				# Check if this is a single instance and has a timestamp in it
				self.timestamp = 0
				self.state = UavTalk.STATE_TIMESTAMP
			else:
				# If there is a payload get it, otherwise receive checksum
				if self.length > 0:
					self.state = UavTalk.STATE_DATA
				else:
					self.state = UavTalk.STATE_CS

			self.rxCount = 0
			self.rxBuffer = ""
			
		elif self.state == UavTalk.STATE_INSTID:
			
			# update the CRC
			self.cs = self.__updateCRC(rxbyte)
			
			self.instId = self.instId + rxbyte << (8*self.rxCount)
			self.rxCount = self.rxCount + 1

			if self.rxCount < 2:
				# wait for both bytes
				return
			
			self.rxCount = 0
			
			# If there is a timestamp, get it
			if self.length > 0 and self.type & UAVTALK_TIMESTAMPED:
				self.timestamp = 0
				self.state = UavTalk.STATE_TIMESTAMP
			elif self.length > 0:
				# If there is a payload get it, otherwise receive checksum
				self.state = UavTalk.STATE_DATA
			else:
				self.state = UavTalk.STATE_CS
			
		elif self.state == UavTalk.STATE_TIMESTAMP:

			# update the CRC
			self.cs = self.__updateCRC(rxbyte)

			self.timestamp = self.timestamp + (rxbyte << (8*self.rxCount))
			self.rxCount = self.rxCount + 1

			if self.rxCount < 2:
				# wait for both bytes
				return

			# Account for the 16 bit limitations of the timestamp
			if self.timestamp < self.lastTimestamp:
				self.timestampBase = self.timestampBase + 65536
			self.lastTimestamp = self.timestamp
			self.timestamp = self.timestamp + self.timestampBase

			self.rxCount = 0

			# If there is a payload get it, otherwise receive checksum
			if self.length > 0:
				self.state = UavTalk.STATE_DATA
			else:
				self.state = UavTalk.STATE_CS
			
		elif self.state == UavTalk.STATE_DATA:
			
			# update the CRC
			self.cs = self.__updateCRC(rxbyte)
			
			self.rxBuffer += chr(rxbyte)
			self.rxCount = self.rxCount + 1

			if self.rxCount < self.length:
				# wait for the rest of the bytes
				return

			self.state = UavTalk.STATE_CS
			self.rxCount = 0

			
		elif self.state == UavTalk.STATE_CS:
			
			# check the CRC byte
			if rxbyte != self.cs:
				print "Bad crc. Got " + hex(rxbyte) + " but predicted " + hex(self.cs)
				self.state = UavTalk.STATE_ERROR
				return
			
			if (self.rxPacketLength != (self.packetSize + 1)):
			   	# packet error - mismatched packet size
			   	print "Bad packet size"
				self.state = UavTalk.STATE_ERROR
				return
			
			self.state = UavTalk.STATE_COMPLETE
			
		else:
			self.state = UavTalk.STATE_ERROR

	def sendSingleObject(self, obj):
		"""
		Generates a string containing a UAVTalk packet describing this object
		"""

		uavo_key = '{0:08x}'.format(obj.uavo_id)
		uavo_def = self.uavo_defs[uavo_key]

		length = 4

		import struct
		if uavo_def.meta['is_single_inst']:
			uavo_hdr_fmt = "<BBHI"
			length = struct.calcsize(uavo_hdr_fmt) + uavo_def.get_size_of_data()
			hdr = struct.pack(uavo_hdr_fmt, UavTalk.SYNC_VAL, UavTalk.TYPE_OBJ | UavTalk.TYPE_VER,  length, obj.uavo_id)
		else:
			uavo_hdr_fmt = "<BBHIH"
			length = struct.calcsize(uavo_hdr_fmt) + uavo_def.get_size_of_data()
			hdr = struct.pack(uavo_hdr_fmt, UavTalk.SYNC_VAL, UavTalk.TYPE_OBJ | UavTalk.TYPE_VER, length, obj.uavo_id, obj.inst_id)

		dat = uavo_def.bytes_from_instance(obj)
		packet = hdr + dat.tostring()

		cs = 0
		for b in packet:
			cs = self.__updateCRC(ord(b), cs)

		packet += chr(cs)

		return packet

	def __updateCRC(self, byte, cs=None):
		"""
		Calculate a CRC consistently with how they are computed on the firmware side
		"""

		# CRC lookup table
		crc_table = [
			0x00, 0x07, 0x0e, 0x09, 0x1c, 0x1b, 0x12, 0x15, 0x38, 0x3f, 0x36, 0x31, 0x24, 0x23, 0x2a, 0x2d,
			0x70, 0x77, 0x7e, 0x79, 0x6c, 0x6b, 0x62, 0x65, 0x48, 0x4f, 0x46, 0x41, 0x54, 0x53, 0x5a, 0x5d,
			0xe0, 0xe7, 0xee, 0xe9, 0xfc, 0xfb, 0xf2, 0xf5, 0xd8, 0xdf, 0xd6, 0xd1, 0xc4, 0xc3, 0xca, 0xcd,
			0x90, 0x97, 0x9e, 0x99, 0x8c, 0x8b, 0x82, 0x85, 0xa8, 0xaf, 0xa6, 0xa1, 0xb4, 0xb3, 0xba, 0xbd,
			0xc7, 0xc0, 0xc9, 0xce, 0xdb, 0xdc, 0xd5, 0xd2, 0xff, 0xf8, 0xf1, 0xf6, 0xe3, 0xe4, 0xed, 0xea,
			0xb7, 0xb0, 0xb9, 0xbe, 0xab, 0xac, 0xa5, 0xa2, 0x8f, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9d, 0x9a,
			0x27, 0x20, 0x29, 0x2e, 0x3b, 0x3c, 0x35, 0x32, 0x1f, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0d, 0x0a,
			0x57, 0x50, 0x59, 0x5e, 0x4b, 0x4c, 0x45, 0x42, 0x6f, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7d, 0x7a,
			0x89, 0x8e, 0x87, 0x80, 0x95, 0x92, 0x9b, 0x9c, 0xb1, 0xb6, 0xbf, 0xb8, 0xad, 0xaa, 0xa3, 0xa4,
			0xf9, 0xfe, 0xf7, 0xf0, 0xe5, 0xe2, 0xeb, 0xec, 0xc1, 0xc6, 0xcf, 0xc8, 0xdd, 0xda, 0xd3, 0xd4,
			0x69, 0x6e, 0x67, 0x60, 0x75, 0x72, 0x7b, 0x7c, 0x51, 0x56, 0x5f, 0x58, 0x4d, 0x4a, 0x43, 0x44,
			0x19, 0x1e, 0x17, 0x10, 0x05, 0x02, 0x0b, 0x0c, 0x21, 0x26, 0x2f, 0x28, 0x3d, 0x3a, 0x33, 0x34,
			0x4e, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5c, 0x5b, 0x76, 0x71, 0x78, 0x7f, 0x6a, 0x6d, 0x64, 0x63,
			0x3e, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2c, 0x2b, 0x06, 0x01, 0x08, 0x0f, 0x1a, 0x1d, 0x14, 0x13,
			0xae, 0xa9, 0xa0, 0xa7, 0xb2, 0xb5, 0xbc, 0xbb, 0x96, 0x91, 0x98, 0x9f, 0x8a, 0x8d, 0x84, 0x83,
			0xde, 0xd9, 0xd0, 0xd7, 0xc2, 0xc5, 0xcc, 0xcb, 0xe6, 0xe1, 0xe8, 0xef, 0xfa, 0xfd, 0xf4, 0xf3
		]

		if cs is None:
			cs = self.cs

		return crc_table[cs ^ byte]

	
