"""
Implements the uavtalk protocol.

Copyright (C) 2014-2015 Tau Labs, http://taulabs.org
Licensed under the GNU LGPL version 2.1 or any later version (see COPYING.LESSER)


Ordinarily one would use the methods exposed by the telemetry module instead of
this interface.
"""

import struct
import time

__all__ = [ "send_object", "process_stream" ]

# Constants used for UAVTalk parsing
(MIN_HEADER_LENGTH, MAX_HEADER_LENGTH, MAX_PAYLOAD_LENGTH) = (8, 12, (256-12))
(SYNC_VAL) = (0x3C)
(TYPE_MASK, TYPE_VER) = (0x78, 0x20)
(TIMESTAMPED) = (0x80)
(TYPE_OBJ, TYPE_OBJ_REQ, TYPE_OBJ_ACK, TYPE_ACK, TYPE_NACK, TYPE_OBJ_TS, TYPE_OBJ_ACK_TS) = (0x00, 0x01, 0x02, 0x03, 0x04, 0x80, 0x82)

# Serialization of header elements

# sync(1) + type(1) + len(2) + objid(4)
header_fmt = struct.Struct("<BBHL")
logheader_fmt = struct.Struct("<IQ")
timestamp_fmt = struct.Struct("<H")
instance_fmt = struct.Struct("<H")

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

def process_stream(uavo_defs, use_walltime=False, gcs_timestamps=False):
    """Generator function that parses uavotalk stream.
    
    You are expected to send more bytes, or '' to it, until EOF.  Then send
    None.  After that, you may continue to receive objects back because of
    buffering."""

    # These are used for accounting for timestamp wraparound
    timestamp_base = 0
    last_timestamp = 0

    received = 0

    buf = ''
    buf_offset = 0

    pending_pieces = []

    while True:
        # If we don't have sufficient data buffered, join up any chunks we've 
        # been given to ensure pending_pieces is empty for the rest of this loop.
        #
        # in other words, don't mix the pending_pieces drain model and the
        # buffer concatenation model within a loop iteration.
        #
        # 10k chosen here to be bigger than any plausible uavo; could calculate
        # this instead from our known uav objects
        if len(buf) - buf_offset < 10240:
            #print "stitch pp=%d"%(len(pending_pieces))
            pending_pieces.insert(0, buf[buf_offset:])
            buf_offset = 0

            buf = ''.join(pending_pieces)
            pending_pieces = []

        if gcs_timestamps:
            while len(buf) < logheader_fmt.size + buf_offset:
                rx = yield None

                if rx is None:
                    return

                buf = buf + rx

            overrideTimestamp, logHdrLen = logheader_fmt.unpack_from(buf,buf_offset)

            buf_offset += logheader_fmt.size

        # Ensure we have enough room for all the
        # plain required fields to avoid duplicating
        # this code lots.
        # sync(1) + type(1) + len(2) + objid(4)

        while (len(buf) < header_fmt.size + buf_offset) or (buf[buf_offset] != chr(SYNC_VAL)):
            #print "waitingsync len=%d, offset=%d"%(len(buf), buf_offset)

            if len(buf) < header_fmt.size + 1 + buf_offset:
                rx = yield None

                if rx is None:
                    #end of stream, stopiteration
                    return

                buf = buf + rx

            for i in xrange(buf_offset, len(buf)):
                if buf[i] == chr(SYNC_VAL):
                    break

            #print "skipping from %d to %d"%(buf_offset, i)

            # Trim off irrelevant stuff, loop and try again
            buf_offset = i

        (sync, pack_type, pack_len, objId) = header_fmt.unpack_from(buf, buf_offset)

        if (pack_type & TYPE_MASK) != TYPE_VER:
            print "badver %x"%(pack_type)
            buf_offset += 1
            continue    # go to top to look for sync
    
        pack_type &= ~ TYPE_MASK
        
        if pack_len < MIN_HEADER_LENGTH or pack_len > MAX_HEADER_LENGTH + MAX_PAYLOAD_LENGTH:
            print "badlen %d"%(pack_len)
            buf_offset += 1
            continue
        
        # Search for object.
        uavo_key = '{0:08x}'.format(objId)
        if not uavo_key in uavo_defs:
            #print "Unknown object 0x" + uavo_key
            obj = None
        else:
            obj = uavo_defs[uavo_key]

        # Determine data length
        if pack_type == TYPE_OBJ_REQ or pack_type == TYPE_ACK or pack_type == TYPE_NACK:
            obj_len = 0
            timestamp_len = 0
            obj = None
        else:
            if obj is not None:
                timestamp_len = timestamp_fmt.size if pack_type == TYPE_OBJ_TS or pack_type == TYPE_OBJ_ACK_TS else 0
                obj_len = obj.get_size_of_data()
            else:
                # we don't know anything, so fudge to keep sync.
                timestamp_len = 0
                obj_len = pack_len - header_fmt.size

        if obj is not None and not obj._single:
            instance_len = 2
        else:
            instance_len = 0

        # Check length and determine next state
        if obj_len >= MAX_PAYLOAD_LENGTH:
            print "bad len-- bad xml?"
            #should never happen; requires invalid uavo xml
            buf_offset += 1
            continue

        # calc_size, AKA timestamp, and obj data
        # as appropriate, plus our current header
        # also equivalent to the offset of the CRC in the packet

        calc_size = header_fmt.size + instance_len + timestamp_len + obj_len

        # Check the lengths match
        if calc_size != pack_len:
            print "mismatched size id=%s %d vs %d, type %d"%(uavo_key,
                calc_size, pack_len, pack_type)

            # packet error - mismatched packet size
            # Consume a byte to try syncing right after where we
            # did...
            buf_offset += 1
            continue

        # OK, at this point we are seriously hoping to receive
        # a packet.  Time for another loop to make sure we have
        # enough data.
        # +1 here is for CRC-8
        while len(buf) < calc_size + 1 + buf_offset:
            rx = yield None

            if rx is None:
                #end of stream, stopiteration
                return

            buf += rx

        # check the CRC byte

        cs = calcCRC(buf[buf_offset:calc_size+buf_offset])
        recv_cs = buf[buf_offset+calc_size]

        if recv_cs != cs:
            print "Bad crc. Got " + hex(ord(recv_cs)) + " but predicted " + hex(ord(cs))

            buf_offset += 1

            continue

        if instance_len:
            instance_id = instance_fmt.unpack_from(buf, header_fmt.size + buf_offset)[0]
        else:
            instance_id = None

        if timestamp_len:
            # pull the timestamp from the packet
            timestamp = timestamp_fmt.unpack_from(buf, header_fmt.size + instance_len + buf_offset)[0]

            # handle wraparound
            if timestamp < last_timestamp:
                timestamp_base = timestamp_base + 65536
            last_timestamp = timestamp
            timestamp += timestamp_base
        else:
            timestamp = last_timestamp

        if use_walltime:
            timestamp = int(time.time()*1000.0)

        if gcs_timestamps:
            timestamp = overrideTimestamp

        if obj is not None:
            offset = header_fmt.size + instance_len + timestamp_len + buf_offset
            objInstance = obj.from_bytes(buf, timestamp, instance_id, offset=offset)
            received += 1
            if not (received % 20000):
                print "received %d objs"%(received)

            next_recv = yield objInstance
        else:
            next_recv = None

        buf_offset += calc_size + 1

        if next_recv is not None and next_recv != '':
            pending_pieces.append(next_recv)

def send_object(obj):
    """Generates a string containing a UAVTalk packet describing this object"""

    hdr = header_fmt.pack(SYNC_VAL, TYPE_OBJ | TYPE_VER,
        header_fmt.size + obj.get_size_of_data(),
        obj._id)

    packet = hdr + obj.to_bytes()

    packet += calcCRC(packet)

    return packet

def request_object(obj):
    """Makes a request for this object"""
    packet = header_fmt.pack(SYNC_VAL, TYPE_OBJ_REQ | TYPE_VER,
        header_fmt.size, obj._id)

    packet += calcCRC(packet)

    return packet

def calcCRC(str):
    """
    Calculate a CRC consistently with how they are computed on the firmware side
    """

    cs = 0

    for c in str:
        cs = crc_table[cs ^ ord(c)]

    return chr(cs)
