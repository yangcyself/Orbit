import socket
import struct

from utils.tcp_utils import send_data, recv_data
from utils.myTypes import myStr, myNumpyArray, myInt, myFloat
import uuid
import numpy as np

class ServerValue:
    """
    A value that is stored on the server, could be lazily retrieved by the client.
    """
    def __init__(self, name, client):
        self.name = name
        self.id = str(uuid.uuid4().int)
        self.value = None
        self.client = client

        self.client._set_value_ref(self.id, self.name)

    def _get_value(self):
        self.value = self.client._get_value(self.id)

    def get_value(self):
        if self.value is None:
            self._get_value()
        return self.value

    def __repr__(self):
        print(f"ServerValue({self.name}, {self.value})")

    def __del__(self):
        self.client._del_value(self.id)

class RobotClient:

    def __init__(self, host, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((host, port))

    def _send(self, msg):
        send_data(self.client, msg)
    
    def _recv(self):
        return recv_data(self.client)

    def __del__(self):
        self.client.close()

    # second layer of abstraction
    def _set_value(self, name, value):
        self._send(myStr("set_value").to_bytes()
                    + myStr(name).to_bytes()
                    + myNumpyArray(value).to_bytes()
        )
    
    def _get_value(self, name):
        self._send(myStr("get_value").to_bytes() + myStr(name).to_bytes())
        recv_data = self._recv()
        return myNumpyArray.from_buffer(recv_data)[0]

    def _set_value_ref(self, name, ref_name):
        self._send(myStr("set_value_ref").to_bytes() 
                    + myStr(name).to_bytes()
                    + myStr(ref_name).to_bytes()
        )

    def _del_value(self, name):
        self._send(myStr("del_value").to_bytes() + myStr(name).to_bytes())

    def _exec_command(self, cmd, count):
        self._send(myStr("exec_command").to_bytes() 
                    + myStr(cmd).to_bytes()
                    + myInt(count).to_bytes()
        )

    # third layer of abstraction
    def get_server_value(self, name):
        return ServerValue(name, self)

    def cmd_moveto(self, target_pos):
        """Robot Action moveto
           - Read: moveto_target_pos
           - Write: None
        Args:
            target_pos (_type_): numpy.ndarray([2,])
        """
        self._set_value("moveto_target_pos", target_pos)
        self._exec_command("moveto", 15)



if __name__ == '__main__':
    client = RobotClient("localhost", 12345)
    import time
    time.sleep(5)
    print("start")
    debug_info = client.get_server_value("obs/debug/debug_info")
    print("debug_info",debug_info.get_value())
    for i in range(100):
        _debug = client.get_server_value("obs/debug/debug_info")
        print(_debug.get_value())
        client.cmd_moveto(np.array([-2 + 1* np.sin(i/50*3.14), -1 + 1 * np.cos(i/50*3.14)]))
        time.sleep(2)
    print("last",debug_info.get_value())


