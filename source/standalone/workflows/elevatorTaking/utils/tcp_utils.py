import socket
import struct
import threading
import copy

def send_data(conn, data):
    # Prefix each message with a 4-byte length (network byte order)
    length_prefix = struct.pack('!I', len(data))
    conn.sendall(length_prefix + data)

def recv_data(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to receive n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


class Racedata(object):
    """The object that holds the data that is shared between the threads.
    """
    def __init__(self):
        # self.lock = threading.Lock()
        self.meta_lock = threading.Lock()
        self.locks = {}
        self.values = {}

        self.command_lock = threading.Lock()
        self.command = None
        self.command_count = 0

    def setdata(self, name, data):
        if name not in self.locks and data is not None:
            self.meta_lock.acquire()
            self.locks[name] = threading.Lock()
            self.values[name] = data
            self.meta_lock.release()
            return 
        if data is None: # delete the data
            self.deldata(name)
            return
        self.locks[name].acquire()
        self.values[name] = data
        self.locks[name].release()

    def setdataref(self, name, refname):
        if name not in self.locks:
            self.meta_lock.acquire()
            self.locks[name] = threading.Lock()
            self.values[name] = self.values[refname]
            self.meta_lock.release()
            return 
        self.locks[name].acquire()
        self.values[name] = self.values[refname]
        self.locks[name].release()

    def getdata(self, name):
        if name not in self.locks:
            return None
        self.locks[name].acquire()
        data = self.values[name]
        self.locks[name].release()
        return data

    def deldata(self, name):
        if name not in self.locks:
            return
        self.meta_lock.acquire()
        del self.locks[name]
        del self.values[name]
        self.meta_lock.release()

    def setcommand(self, command, count):
        self.command_lock.acquire()
        self.command = command
        self.command_count = count
        self.command_lock.release()

    def getcommand(self):
        self.command_lock.acquire()
        command = self.command
        count = self.command_count
        self.command_lock.release()
        return command, count

    def popcommand(self):
        # get command and count, set command to None
        self.command_lock.acquire()
        command = self.command
        count = self.command_count
        self.command = None
        self.command_lock.release()
        return command, count

    def command_count_dec(self):
        self.command_lock.acquire()
        self.command_count -= 1
        command_count = self.command_count
        self.command_lock.release()
        return command_count

    def reset(self):
        self.meta_lock.acquire()
        self.locks = {}
        self.values = {}
        self.meta_lock.release()

        self.command_lock.acquire()
        self.command = None
        self.command_count = 0
        self.command_lock.release()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_warning(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)

def print_error(msg):
    print(bcolors.FAIL + msg + bcolors.ENDC)
