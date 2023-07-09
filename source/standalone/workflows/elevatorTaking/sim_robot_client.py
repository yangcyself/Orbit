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
        if(isinstance(ref_name, ServerValue)):
            ref_name = ref_name.id
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

    def cmd_pushbtn(self, goal_dof, goal_base_rgb, goal_base_semantic, goal_hand_rgb, goal_hand_semantic):
        """Robot Action pushbtn
           - Read: 
             - pushbtn_goal_dof_pos, 
             - pushbtn_goal_base_rgb, 
             - pushbtn_goal_base_semantic
             - pushbtn_goal_hand_rgb
             - pushbtn_goal_hand_semantic
           - Write: None
        Args:
            goal_dof (ServerValue)
            goal_base_rgb (ServerValue)
            goal_base_semantic (numpy.ndarray(uint8[ W,H,2]))
            goal_hand_rgb (ServerValue)
            goal_hand_semantic (numpy.ndarray(uint8[ W,H,2]))
        """
        self._set_value_ref("pushbtn_goal_dof_pos", goal_dof)
        self._set_value_ref("pushbtn_goal_base_rgb", goal_base_rgb)
        self._set_value("pushbtn_goal_base_semantic", goal_base_semantic)
        self._set_value_ref("pushbtn_goal_hand_rgb", goal_hand_rgb)
        self._set_value("pushbtn_goal_hand_semantic", goal_hand_semantic)
        self._exec_command("pushbtn", 100)


if __name__ == '__main__':
    client = RobotClient("localhost", 12345)
    import time
    time.sleep(5)
    print("start")
    pushbtn_goal_dof = client.get_server_value("obs/low_dim/dof_pos_obsframe")
    pushbtn_goal_base_rgb = client.get_server_value("obs/rgb/base_camera_rgb")
    pushbtn_goal_base_semantic = client.get_server_value("obs/semantic/base_camera_semantic")
    pushbtn_goal_hand_rgb = client.get_server_value("obs/rgb/hand_camera_rgb")
    pushbtn_goal_hand_semantic = client.get_server_value("obs/semantic/hand_camera_semantic")

    client.cmd_moveto(np.array([0.2 , -0.1]))

    goal_base_semantic = pushbtn_goal_base_semantic.get_value()
    goal_hand_semantic = pushbtn_goal_hand_semantic.get_value()
    client.cmd_pushbtn(pushbtn_goal_dof, 
        pushbtn_goal_base_rgb, 
        goal_base_semantic, 
        pushbtn_goal_hand_rgb, 
        goal_hand_semantic)

    import matplotlib.pyplot as plt
    plt.imshow(pushbtn_goal_base_rgb.get_value()[0])
    plt.figure()
    plt.imshow(goal_base_semantic[0,:,:,0])
    plt.figure()
    plt.imshow(pushbtn_goal_hand_rgb.get_value()[0])
    plt.figure()
    plt.imshow(goal_hand_semantic[0,:,:,0])
    plt.show()

