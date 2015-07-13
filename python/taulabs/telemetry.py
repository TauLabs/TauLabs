"""
Interface to telemetry streams -- log, network, or serial.

Copyright (C) 2014-2015 Tau Labs, http://taulabs.org
Licensed under the GNU LGPL version 2.1 or any later version (see COPYING.LESSER)
"""

import socket
import time
import array
import select
import errno

import threading

import uavtalk, uavo_collection, uavo

import os

from abc import ABCMeta, abstractmethod

class TelemetryBase():
    """
    Basic (abstract) implementation of telemetry used by all stream types.
    """

    # This is not a complete implemention / must subclass
    __metaclass__ = ABCMeta

    def __init__(self, githash=None, service_in_iter=True,
            iter_blocks=True, use_walltime=True, do_handshaking=False,
            gcs_timestamps=False, name=None):
        """Instantiates a telemetry instance.  Called only by derived classes.
        
         - githash: revision control id of the UAVO's used to communicate.
             if unspecified, we use the version in this source tree.
         - service_in_iter: whether the actual iterator should service the
             connection/read the file.  If you don't specify this, you probably
             want to call start_thread()
         - iter_blocks: whether the iterator should block for new data.  Only
             really meaningful for network/serial
         - use_walltime: if true, automatically place current time into packets
             received.
         - do_handshaking: if true, indicates this is an interactive stream
             where we should speak the UAVO_GCSTelemetryStats connection status
             protocol.
         - gcs_timestamps: if true, this means we are reading from a file with
             the GCS timestamp protocol.
         - name: a filename to store into .filename for legacy purposes
        """

        uavo_defs = uavo_collection.UAVOCollection()

        if githash:
            uavo_defs.from_git_hash(githash)
        else:
            xml_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                    "shared", "uavobjectdefinition")
            uavo_defs.from_uavo_xml_path(xml_path)

        self.githash = githash

        self.uavo_defs = uavo_defs
        self.uavtalk_generator = uavtalk.process_stream(uavo_defs,
            use_walltime=use_walltime, gcs_timestamps=gcs_timestamps)

        self.uavtalk_generator.send(None)

        self.uavo_list = []

        self.last_values = {}

        self.cond = threading.Condition()

        self.service_in_iter = service_in_iter
        self.iter_blocks = iter_blocks

        self.do_handshaking = do_handshaking
        self.filename = name

        self.eof = False

    def as_numpy_array(self, match_class):
        """ Transforms all received instances of a given object to a numpy array.
        
        match_class: the UAVO_* class you'd like to match.
        """

        import numpy as np

        # Find the subset of this list that is of the requested class
        filtered_list = filter(lambda x: isinstance(x, match_class), self)
 
        # Check for an empty list
        if filtered_list == []:
            return np.array([])
 
        return np.array(filtered_list, dtype=match_class._dtype)

    def __iter__(self):
        """ Iterator service routine. """
        iterIdx = 0

        self.cond.acquire()

        with self.cond:
            while True:
                if iterIdx < len(self.uavo_list):
                    obj = self.uavo_list[iterIdx]

                    iterIdx += 1

                    self.cond.release()
                    try:
                        yield obj
                    finally:
                        self.cond.acquire()
                elif self.iter_blocks and not self._done():
                    if self.service_in_iter:
                        self.cond.release()

                        try:
                            self.service_connection()
                        finally:
                            self.cond.acquire()
                    else:
                        # wait for another thread to fill it in
                        self.cond.wait()
                else:
                    # Don't really recommend this mode anymore/maybe remove
                    if self.service_in_iter and not self._done():
                        # Do at least one non-blocking attempt
                        self.cond.release()

                        try:
                            self.service_connection(0)
                        finally:
                            self.cond.acquire()

                    # I think this should probably keep the index so that
                    # new iterations pick up where we were.. XXX TODO
                    # takes some thought as to what is "right"
                    if iterIdx >= len(self.uavo_list):
                        break

    def __make_handshake(self, handshake):
        return uavo.UAVO_GCSTelemetryStats._make_to_send(0, 0, 0, 0, 0, handshake)

    def __handle_handshake(self, obj):
        if obj.name == "UAVO_FlightTelemetryStats":
            # Handle the telemetry handshaking

            (DISCONNECTED, HANDSHAKE_REQ, HANDSHAKE_ACK, CONNECTED) = (0,1,2,3)

            if obj.Status == DISCONNECTED:
                # Request handshake
                print "Disconnected"
                send_obj = self.__make_handshake(HANDSHAKE_REQ)
            elif obj.Status == HANDSHAKE_ACK:
                # Say connected
                print "Handshake ackd"
                send_obj = self.__make_handshake(CONNECTED)
            elif obj.Status == CONNECTED:
                print "Connected"
                send_obj = self.__make_handshake(CONNECTED)

            self._send(uavtalk.send_object(send_obj))

    def request_object(self, obj):
        if not self.do_handshaking:
            raise ValueError("Can only request on handshaking/bidir sessions")

        self._send(uavtalk.request_object(obj))

    def __handle_frames(self, frames):
        objs = []

        obj = self.uavtalk_generator.send(frames)

        while obj:
            if self.do_handshaking:
                self.__handle_handshake(obj)

            objs.append(obj)

            obj = self.uavtalk_generator.send('')

        # Only traverse the lock when we've processed everything in this
        # batch.
        with self.cond:
            # keep everything in ram forever
            # for now-- in case we wanna see
            self.uavo_list.extend(objs)

            if frames == '':
                self.eof=True

            for obj in objs:
                self.last_values[obj.__class__]=obj

            self.cond.notifyAll()

    def get_last_values(self):
        """ Returns the last instance of each kind of object received. """
        with self.cond:
            return self.last_values.copy()

    def start_thread(self):
        """ Starts a separate thread to service this telemetry connection. """
        if self.service_in_iter:
            # TODO sane exceptions here.
            raise

        if self._done():
            raise

        from threading import Thread

        def run():
            while not self._done():
                self.service_connection()

        t = Thread(target=run, name="telemetry svc thread")

        t.daemon=True

        t.start()

    def service_connection(self, timeout=None):
        """
        Receive and parse data from a connection and handle the basic
        handshaking with the flight controller
        """

        if timeout is not None:
            finish_time = time.time() + timeout
        else:
            finish_time = None

        data = self._receive(finish_time)
        self.__handle_frames(data)

    @abstractmethod
    def _receive(self, finish_time):
        return

    # No implementation required, so not abstract
    def _send(self, msg):
        return

    def _done(self):
        with self.cond:
            return self.eof

class FDTelemetry(TelemetryBase):
    """
    Implementation of bidirectional telemetry from a file descriptor.
    
    Intended for serial and network streams.
    """

    def __init__(self, fd, *args, **kwargs):
        """ Instantiates a telemetry instance on a given fd.
        
        Probably should only be called by derived classes.
        
         - fd: the file descriptor to perform telemetry operations upon
        
        Meaningful parameters passed up to TelemetryBase include: githash,
        service_in_iter, iter_blocks, use_walltime
        """

        TelemetryBase.__init__(self, do_handshaking=True,
                gcs_timestamps=False,  *args, **kwargs)

        self.recv_buf = ''
        self.send_buf = ''

        self.fd = fd

    def _receive(self, finish_time):
        """ Fetch available data from file descriptor. """

        # Always do some minimal IO if possible
        self._do_io(0)

        while (len(self.recv_buf) < 1) and self._do_io(finish_time):
            pass

        if len(self.recv_buf) < 1:
            return None

        ret = self.recv_buf
        self.recv_buf = ''

        return ret

    # Call select and do one set of IO operations.
    def _do_io(self, finish_time):
        rdSet = []
        wrSet = []

        didStuff = False

        if len(self.recv_buf) < 1024:
            rdSet.append(self.fd)

        if len(self.send_buf) > 0:
            wrSet.append(self.fd)

        now = time.time()
        if finish_time is None:
            r,w,e = select.select(rdSet, wrSet, [])
        else:
            tm = finish_time-now
            if tm < 0: tm=0

            r,w,e = select.select(rdSet, wrSet, [], tm)

        if r:
            # Shouldn't throw an exception-- they just told us
            # it was ready for read.
            chunk = os.read(self.fd, 1024)
            if chunk == '':
                raise RuntimeError("stream closed")

            self.recv_buf = self.recv_buf + chunk

            didStuff = True

        if w:
            written = os.write(self.fd, self.send_buf)

            if written > 0:
                self.send_buf = self.send_buf[written:]

            didStuff = True

        return didStuff

    def _send(self, msg):
        """ Send a string out the TCP socket """

        self.send_buf += msg

        self._do_io(0)

class NetworkTelemetry(FDTelemetry):
    """ TCP telemetry interface. """
    def __init__(self, host="127.0.0.1", port=9000, *args, **kwargs):
        """ Creates a telemetry instance talking over TCP.
        
         - host: hostname to connect to (default localhost)
         - port: port number to communicate on (default 9000)

        Meaningful parameters passed up to TelemetryBase include: githash,
        service_in_iter, iter_blocks, use_walltime
        """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))

        s.setblocking(0)

        self.sock = s

        FDTelemetry.__init__(self, fd=s.fileno(), *args, **kwargs)

# TODO XXX : Plumb appropriate cleanup / file close for these classes

class SerialTelemetry(FDTelemetry):
    """ Serial telemetry interface """
    def __init__(self, port, speed=115200, *args, **kwargs):
        """ Creates telemetry instance talking over (real or virtual) serial port.
        
         - port: Serial port path
         - speed: Baud rate (doesn't really matter for VCP, defaults 115200)

        Meaningful parameters passed up to TelemetryBase include: githash,
        service_in_iter, iter_blocks, use_walltime
        
        Requires the module pyserial to provide OS independance.
        """

        import serial

        ser = serial.Serial(port, speed)

        FDTelemetry.__init__(self, fd=ser.fileno(), *args, **kwargs)

class FileTelemetry(TelemetryBase):
    """ Telemetry interface to data in a file """

    def __init__(self, file_obj, parse_header=False,
             *args, **kwargs):
        """ Instantiates a telemetry instance reading from a file.
        
         - file_obj: the file object to read from
         - parse_header: whether to read a header like the GCS writes from the
           file.

        Meaningful parameters passed up to TelemetryBase include: githash,
        service_in_iter, iter_blocks, gcs_timestamps
        """

        self.f = file_obj

        if parse_header:
            # Check the header signature
            #    First line is "Tau Labs git hash:"
            #    Second line is the actual git hash
            #    Third line is the UAVO hash
            #    Fourth line is "##"
            sig = self.f.readline()
            if sig != 'Tau Labs git hash:\n':
                print "Source file does not have a recognized header signature"
                print '|' + sig + '|'
                raise IOError("no header signature")
            # Determine the git hash that this log file is based on
            githash = self.f.readline()[:-1]
            if githash.find(':') != -1:
                import re
                githash = re.search(':(\w*)\W', githash).group(1)

            print "Log file is based on git hash: %s" % githash

            uavohash = self.f.readline()
            divider = self.f.readline()

            TelemetryBase.__init__(self, service_in_iter=False, iter_blocks=True,
                do_handshaking=False, githash=githash, use_walltime=False,
                *args, **kwargs)
        else:
            TelemetryBase.__init__(self, service_in_iter=False, iter_blocks=True,
                do_handshaking=False, use_walltime=False, *args, **kwargs)

        self.done=False
        self.start_thread()

    def _receive(self, finish_time):
        """ Fetch available data from file """

        buf = self.f.read(524288)   # 512k

        return buf

def get_telemetry_by_args(desc="Process telemetry"):
    """ Parses command line to decide how to get a telemetry object. """
    # Setup the command line arguments.
    import argparse
    parser = argparse.ArgumentParser(description=desc)

    # Log format indicates this log is using the old file format which
    # embeds the timestamping information between the UAVTalk packet
    # instead of as part of the packet
    parser.add_argument("-t", "--timestamped",
                        action  = 'store_false',
                        default = True,
                        help    = "indicate that this is not timestamped in GCS format")

    parser.add_argument("-g", "--githash",
                        action  = "store",
                        dest    = "githash",
                        help    = "override githash for UAVO XML definitions")

    parser.add_argument("-s", "--serial",
                        action  = "store_true",
                        default = False,
                        dest    = "serial",
                        help    = "indicates that source is a serial port")

    parser.add_argument("-b", "--baudrate",
                        action  = "store",
                        dest    = "baud",
                        help    = "baud rate for serial communications")

    parser.add_argument("source",
                        help  = "file, host:port, or serial port to get telemetry from")

    # Parse the command-line.
    args = parser.parse_args()

    parse_header = False
    githash = None

    if args.githash is not None:
        # If we specify the log header no need to attempt to parse it
        githash = args.githash
    else:
        parse_header = True # only for files

    from taulabs import telemetry

    if args.serial:
        return telemetry.SerialTelemetry(args.source, speed=args.baud)

    if args.baud is not None:
        parser.print_help()
        raise ValueError("Baud rates only apply to serial ports")

    import os.path

    if os.path.isfile(args.source):
        file_obj = file(args.source, 'r')

        t = telemetry.FileTelemetry(file_obj, parse_header=parse_header,
            gcs_timestamps=args.timestamped, name=args.source)

        return t

    # OK, running out of options, time to try the network!
    host,sep,port = args.source.partition(':')

    if sep != ':':
        parser.print_help()
        raise ValueError("Target doesn't exist and isn't a network address")

    return telemetry.NetworkTelemetry(host=host, port=int(port), name=args.source)
