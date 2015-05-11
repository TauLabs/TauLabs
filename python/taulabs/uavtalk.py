import struct
import time

# Constants used for UAVTalk parsing
(MIN_HEADER_LENGTH, MAX_HEADER_LENGTH, MAX_PAYLOAD_LENGTH) = (8, 12, (256-12))
(SYNC_VAL) = (0x3C)
(TYPE_MASK, TYPE_VER) = (0x78, 0x20)
(TIMESTAMPED) = (0x80)
(TYPE_OBJ, TYPE_OBJ_REQ, TYPE_OBJ_ACK, TYPE_ACK, TYPE_NACK, TYPE_OBJ_TS, TYPE_OBJ_ACK_TS) = (0x00, 0x01, 0x02, 0x03, 0x04, 0x80, 0x82)

# sync(1) + type(1) + len(2) + objid(4) 
headerFmt = struct.Struct("<BBHL")
logHeaderFmt = struct.Struct("<IQ")
timestampFmt = struct.Struct("<H") 

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

def processStream(uavo_defs, useWallTime=False, logTimestamps=False):
    # These are used for accounting for timestamp wraparound
    timestampBase = 0
    lastTimestamp = 0

    received = 0

    packetBytes = ''

    while True:
        if logTimestamps:
            while len(packetBytes) < logHeaderFmt.size:
                rx = yield None

                if rx is None:
                    return

                packetBytes = packetBytes + rx

            overrideTimestamp, logHdrLen = logHeaderFmt.unpack_from(packetBytes,0)

            packetBytes = packetBytes[logHeaderFmt.size:]

        # Ensure we have enough room for all the
        # plain required fields to avoid duplicating
        # this code lots.
        # sync(1) + type(1) + len(2) + objid(4) 

        while (len(packetBytes) < headerFmt.size) or (packetBytes[0] != chr(SYNC_VAL)):
            #print "waitingsync len=%d"%(len(packetBytes))

            rx = yield None

            if rx is None:
                #end of stream, stopiteration
                return

            packetBytes = packetBytes + rx

            for i in xrange(len(packetBytes)):
                if packetBytes[i] == chr(SYNC_VAL):
                    break

            # Trim off irrelevant stuff, loop and try again
            packetBytes = packetBytes[i:]

        (sync, packetType, packetLen, objId) = headerFmt.unpack_from(packetBytes,0)

        if (packetType & TYPE_MASK) != TYPE_VER:
            print "badver %x"%(packetType)
            packetBytes = packetBytes[1:]
            continue    # go to top to look for sync
    
        packetType &= ~ TYPE_MASK
        
        if packetLen < MIN_HEADER_LENGTH or packetLen > MAX_HEADER_LENGTH + MAX_PAYLOAD_LENGTH:
            print "badlen %d"%(packetLen)
            packetBytes = packetBytes[1:]
            continue
        
        # Search for object.
        uavo_key = '{0:08x}'.format(objId)
        if not uavo_key in uavo_defs:
            #print "Unknown object 0x" + uavo_key
            obj = None
        else:
            obj = uavo_defs[uavo_key]

        # Before there used to be code to handle instance offsetting
        # here, but it looks like it moved (rightfully) into the object
            
        # Determine data length
        if packetType == TYPE_OBJ_REQ or packetType == TYPE_ACK or packetType == TYPE_NACK:
            objLength = 0
            timestampLength = 0
        else:
            if obj is not None:
                timestampLength = timestampFmt.size if packetType == TYPE_OBJ_TS or packetType == TYPE_OBJ_ACK_TS else 0
                objLength = obj.get_size_of_data()
            else:
                # we don't know anything, so fudge to keep sync.
                timestampLength = 0
                objLength = packetLen - headerFmt.size

        # Check length and determine next state
        if objLength >= MAX_PAYLOAD_LENGTH:
            print "bad len-- bad xml?"
            #should never happen; requires invalid uavo xml
            packetBytes = packetBytes[1:]
            continue

        # calcedSize, AKA timestamp, and obj data
        # as appropriate, plus our current header
        # also equivalent to the offset of the CRC in the packet

        calcedSize = headerFmt.size + timestampLength + objLength

        # Check the lengths match
        if calcedSize != packetLen:
            print "mismatched size id=%s %d vs %d, type %d"%(uavo_key,
                calcedSize, packetLen, packetType)

            # packet error - mismatched packet size
            # Consume a byte to try syncing right after where we
            # did...
            packetBytes = packetBytes[1:]
            continue

        # OK, at this point we are seriously hoping to receive
        # a packet.  Time for another loop to make sure we have
        # enough data.
        # +1 here is for CRC-8
        while len(packetBytes) < calcedSize + 1:
            rx = yield None

            if rx is None:
                #end of stream, stopiteration
                return

            packetBytes += rx

        cs = __calcCRC(packetBytes[0:calcedSize])
        
        # check the CRC byte

        recvcs = ord(packetBytes[calcedSize])

        if recvcs != cs:
            print "Bad crc. Got " + hex(recvcs) + " but predicted " + hex(cs)

            packetBytes = packetBytes[1:]
            continue
        
        if timestampLength:
            # pull the timestamp from the packet
	    timestamp = timestampFmt.unpack_from(packetBytes, headerFmt.size)[0]

            # handle wraparound
            if timestamp < lastTimestamp:
                timestampBase = timestampBase + 65536
            lastTimestamp = timestamp
            timestamp += timestampBase
        else:
            timestamp = lastTimestamp

        if useWallTime:
            timestamp = int(time.time()*1000.0)

        if logTimestamps:
            timestamp = overrideTimestamp

        if obj is not None:
            objInstance = obj.instance_from_bytes(packetBytes,
                timestamp,
                startOffs = headerFmt.size + timestampLength) 

            received += 1
            if not (received % 20000):
                print "received %d objs"%(received)

            nextRecv = yield objInstance
        else:
            nextRecv = None
        
        packetBytes = packetBytes[calcedSize+1:] 

        if nextRecv is not None:
            packetBytes += nextRecv

def sendSingleObject(obj):
    """
    Generates a string containing a UAVTalk packet describing this object
    """

    uavo_def = obj.uavometa

    hdr = headerFmt.pack(SYNC_VAL, TYPE_OBJ | TYPE_VER,
        headerFmt.size + uavo_def.get_size_of_data(),
        obj.uavo_id)

    packet = hdr + obj.bytes()

    packet += chr(__calcCRC(packet))

    return packet

def __calcCRC(str):
    """
    Calculate a CRC consistently with how they are computed on the firmware side
    """

    cs = 0

    for c in str:
        cs = crc_table[cs ^ ord(c)]

    return cs


