import socket
import taulabs
import time
import array
import select
import errno

import threading

import uavtalk, uavo_collection

class Telemetry():
    """
    Provides a basic telemetry connection to a flight controller
    """

    def __init__(self, uavo_defs=None, githash=None, serviceInIter=True,
            iterBlocks=True, useWallTime=True):
        self.sock = None

        if uavo_defs is None:
            uavo_defs = uavo_collection.UAVOCollection()

            if githash:
                uavo_defs.from_git_hash(githash)
            else:
                uavo_defs.from_uavo_xml_path("../shared/uavobjectdefinition")

        self.uavo_defs = uavo_defs
        self.uavtalk_generator = uavtalk.processStream(uavo_defs,
            useWallTime=useWallTime)

        self.uavtalk_generator.send(None)

        self.gcs_telemetry = {v: k for k, v in self.uavo_defs.items() if v.meta['name']=="GCSTelemetryStats"}.items()[0][0]

        self.recv_buf = ''
        self.send_buf = ''

        self.uavo_list = taulabs.uavo_list.UAVOList(self.uavo_defs)

        self.last_values = {}

        self.cond = threading.Condition()

        self.serviceInIter = serviceInIter
        self.iterBlocks = iterBlocks
        self.iterIdx = 0

    def __iter__(self):
        with self.cond:
            while True:
                if self.iterIdx < len(self.uavo_list):
                    self.cond.release()
                    yield self.uavo_list[self.iterIdx]
                    self.iterIdx += 1
                    self.cond.acquire()
                elif self.iterBlocks and self.sock:
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
                    if self.serviceInIter and self.sock:
                        # Do at least one non-blocking attempt
                        self.cond.release()

                        try:
                            self.serviceConnection(0)
                        finally:
                            self.cond.acquire()

                    if self.iterIdx >= len(self.uavo_list):
                        break

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

    def __handleHandshake(self, obj):
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
            packet = uavtalk.sendSingleObject(send_obj)
            self.__send(packet)


    def __handleFrames(self, frames):
        objs = []

        obj = self.uavtalk_generator.send(frames)

        while obj:
            self.__handleHandshake(obj)

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

        if not self.sock:
            raise

        from threading import Thread

        def run():
            while self.sock:
                self.serviceConnection()    

        t = Thread(target=run, name="telemetry svc thread")

        t.daemon=True

        t.start()

    def serviceConnection(self, timeout=None):
        """
        Receive and parse data from a network connection and handle the basic
        handshaking with the flight controller
        """

        if timeout is not None:
            finishTime = time.time()+timeout
        else:
            finishTime = None

        data = self.__receive(finishTime)
        self.__handleFrames(data)

    def __send(self, msg):
        """ Send a string out the TCP socket """

        self.send_buf += msg

        self.__select(0)

    # Call select and do one set of IO operations.
    def __select(self, finishTime):
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

    def __receive(self, finishTime):
        """ Fetch available data from TCP socket """

        # Always do some minimal IO if possible
        self.__select(0)

        while (len(self.recv_buf) < 1) and self.__select(finishTime):
            pass

        if len(self.recv_buf) < 1:
            return None

        ret=self.recv_buf
        self.recv_buf=''

        return ret

