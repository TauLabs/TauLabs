import socket
import taulabs
import time
import array
import select
import errno

import threading

import uavtalk, uavo_collection

from abc import ABCMeta, abstractmethod

class TelemetryBase():
    """
    Provides a basic telemetry connection to a flight controller
    """

    # This is not a complete implemention / must subclass
    __metaclass__ = ABCMeta

    def __init__(self, uavo_defs=None, githash=None, serviceInIter=True,
            iterBlocks=True, useWallTime=True, doHandshaking=False,
	    weirdTimestamps=False):
        if uavo_defs is None:
            uavo_defs = uavo_collection.UAVOCollection()

            if githash:
                uavo_defs.from_git_hash(githash)
            else:
                uavo_defs.from_uavo_xml_path("shared/uavobjectdefinition")

        self.uavo_defs = uavo_defs
        self.uavtalk_generator = uavtalk.processStream(uavo_defs,
            useWallTime=useWallTime, logTimestamps=weirdTimestamps)

        self.uavtalk_generator.send(None)

        self.gcs_telemetry = {v: k for k, v in self.uavo_defs.items() if v.meta['name']=="GCSTelemetryStats"}.items()[0][0]

        self.uavo_list = taulabs.uavo_list.UAVOList(self.uavo_defs)

        self.last_values = {}

        self.cond = threading.Condition()

        self.serviceInIter = serviceInIter
        self.iterBlocks = iterBlocks

        self.doHandshaking = doHandshaking

    def as_numpy_array(self, match_class): 
        import numpy as np

        # Find the subset of this list that is of the requested class 
        filtered_list = filter(lambda x: isinstance(x, match_class), self) 
 
        # Check for an empty list 
        if filtered_list == []: 
            return np.array([]) 
 
        # Find the uavo definition associated with this UAVO type 
        if not "{0:08x}".format(filtered_list[0].uavo_id) in self.uavo_defs: 
            dtype = None 
        else: 
            uavo_def = self.uavo_defs["{0:08x}".format(filtered_list[0].uavo_id)] 
            dtype  = [('name', 'S20'), ('time', 'double'), ('uavo_id', 'uint')] 
 
            for f in uavo_def.fields: 
                dtype += [(f['name'], '(' + `f['elements']` + ",)" + uavo_def.type_numpy_map[f['type']])] 
 
        return np.array(filtered_list, dtype=dtype) 

    def __iter__(self):
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
                elif self.iterBlocks and not self._done():
                    if self.serviceInIter:
                        self.cond.release()

                        try:
                            self.serviceConnection()
                        finally:
                            self.cond.acquire()
                    else:
                        # wait for another thread to fill it in
                        self.cond.wait()
                else:
                    # Don't really recommend this mode anymore/maybe remove
                    if self.serviceInIter and not self._done():
                        # Do at least one non-blocking attempt
                        self.cond.release()

                        try:
                            self.serviceConnection(0)
                        finally:
                            self.cond.acquire()

                    if iterIdx >= len(self.uavo_list):
                        break

    def _handleHandshake(self, obj):
        if obj.name == "FlightTelemetryStats":
            # Handle the telemetry handshaking

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
            packet = uavtalk.sendSingleObject(send_obj)
            self._send(packet)

    def _handleFrames(self, frames):
        objs = []

        obj = self.uavtalk_generator.send(frames)

        while obj:
            if self.doHandshaking:
                self._handleHandshake(obj)

            objs.append(obj)

            obj = self.uavtalk_generator.send('')

        # Only traverse the lock when we've processed everything in this
        # batch.
        with self.cond:
            # keep everything in ram forever
            # for now-- in case we wanna see
            self.uavo_list.extend(objs)

            for obj in objs:
                self.last_values[obj.name]=obj

            self.cond.notifyAll()

    def get_last_values(self):
        with self.cond:
            return self.last_values.copy()

    def start_thread(self):
        if self.serviceInIter:
            raise

        if self._done():
            raise

        from threading import Thread

        def run():
            while not self._done():
                self.serviceConnection()    

        t = Thread(target=run, name="telemetry svc thread")

        t.daemon=True

        t.start()

    def serviceConnection(self, timeout=None):
        """
        Receive and parse data from a connection and handle the basic
        handshaking with the flight controller
        """

        if timeout is not None:
            finishTime = time.time()+timeout
        else:
            finishTime = None

        data = self._receive(finishTime)
        self._handleFrames(data)

    @abstractmethod
    def _receive(self, finishTime):
        return

    # No implementation required, so not abstract
    def _send(self, msg):
	return

    @abstractmethod
    def _done(self):
        return True

class NetworkTelemetry(TelemetryBase):
    def __init__(self, *args, **kwargs):
        TelemetryBase.__init__(self, doHandshaking=True,
                weirdTimestamps=False,  *args, **kwargs)

        self.recv_buf = ''
        self.send_buf = ''

	self.sock = None

    def open_network(self, host="127.0.0.1", port=9000):
        """ Open a socket on localhost port 9000 """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))

        s.setblocking(0)

        self.sock = s

    def close_network(self):
        """ Close network socket """

        self.sock.close()
        self.sock = None

    def _receive(self, finishTime):
        """ Fetch available data from TCP socket """

        # Always do some minimal IO if possible
        self._do_io(0)

        while (len(self.recv_buf) < 1) and self.do_io(finishTime):
            pass

        if len(self.recv_buf) < 1:
            return None

        ret=self.recv_buf
        self.recv_buf=''

        return ret

    # Call select and do one set of IO operations.
    def _do_io(self, finishTime):
        rdSet = []
        wrSet = []

        didStuff = False

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

            self.recv_buf = self.recv_buf + chunk

            didStuff = True

        if w:
            written = self.sock.send(self.send_buf)
            self.send_buf = self.send_buf[written:]

            didStuff = True

        return didStuff

    def _send(self, msg):
        """ Send a string out the TCP socket """

        self.send_buf += msg

        self._do_io(0)

    def _done(self):
        return self.sock is None

class FileTelemetry(TelemetryBase):
    def __init__(self, filename='sim_log.tll', parseHeader=False,
             *args, **kwargs):
        self.f = file(filename, 'r')

        if parseHeader:
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

            TelemetryBase.__init__(self, serviceInIter=False, iterBlocks=True,
                doHandshaking=False, githash=githash, *args, **kwargs)
        else:
            TelemetryBase.__init__(self, serviceInIter=False, iterBlocks=True,
                doHandshaking=False, *args, **kwargs)

        self.done=False
        self.start_thread()

    def _receive(self, finishTime):
        """ Fetch available data from TCP socket """

        buf = self.f.read(128)

        if buf == '':
            self.done=True

        return buf

    def _done(self):
        return self.done
