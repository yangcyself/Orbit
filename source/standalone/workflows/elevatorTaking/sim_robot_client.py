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

    def _get_command(self):
        self._send(myStr("get_command").to_bytes())
        recv_data = self._recv()
        cmd,l = myStr.from_buffer(recv_data)
        count, _ = myInt.from_buffer(recv_data[l:])
        return cmd, count

    # third layer of abstraction
    def get_server_value(self, name):
        return ServerValue(name, self)

    def cmd_base(self, task_frame_shift):
        if task_frame_shift is None:
            self._set_value("base_task_frame_shift", np.zeros((1,3)))
        elif isinstance(task_frame_shift, np.ndarray): 
            self._set_value("base_task_frame_shift", task_frame_shift)
        else:
            self._set_value_ref("base_task_frame_shift", task_frame_shift)

    def cmd_moveto(self, target_pos, task_frame_shift=None):
        """Robot Action moveto
           - Read: 
             - base_task_frame_shift
             - moveto_target_pos
           - Write: None
        Args:
            task_frame_shift: can be None, np.ndarray or ServerValue
            target_pos (_type_): numpy.ndarray([2,])
        """
        self.cmd_base(task_frame_shift)

        if isinstance(target_pos, np.ndarray): 
            self._set_value("moveto_target_pos", target_pos)
        else:
            self._set_value_ref("moveto_target_pos", target_pos)

        self._exec_command("moveto", 15)

    def cmd_robomimicBase(self, goal_dof, goal_base_rgb, goal_base_semantic, goal_hand_rgb, goal_hand_semantic, task_frame_shift=None):
        """Robot Action pushbtn
           - Read: 
             - base_task_frame_shift
             - mimic_goal_dof_pos, 
             - mimic_goal_base_rgb, 
             - mimic_goal_base_semantic
             - mimic_goal_hand_rgb
             - mimic_goal_hand_semantic
           - Write: None
        Args:
            goal_dof (ServerValue)
            goal_base_rgb (ServerValue)
            goal_base_semantic (numpy.ndarray(uint8[ W,H,2]))
            goal_hand_rgb (ServerValue)
            goal_hand_semantic (numpy.ndarray(uint8[ W,H,2]))
        """
        self.cmd_base(task_frame_shift)
        self._set_value_ref("mimic_goal_dof_pos", goal_dof)
        self._set_value_ref("mimic_goal_base_rgb", goal_base_rgb)
        self._set_value("mimic_goal_base_semantic", goal_base_semantic)
        self._set_value_ref("mimic_goal_hand_rgb", goal_hand_rgb)
        self._set_value("mimic_goal_hand_semantic", goal_hand_semantic)


    def cmd_pushBtn(self, goal_dof, goal_base_rgb, goal_base_semantic, goal_hand_rgb, goal_hand_semantic, task_frame_shift=None):
        self.cmd_robomimicBase(goal_dof, goal_base_rgb, goal_base_semantic, goal_hand_rgb, goal_hand_semantic, task_frame_shift)
        self._exec_command("pushBtn", 100)

    def cmd_movetoBtn(self, goal_dof, goal_base_rgb, goal_base_semantic, goal_hand_rgb, goal_hand_semantic, task_frame_shift=None):
        self.cmd_robomimicBase(goal_dof, goal_base_rgb, goal_base_semantic, goal_hand_rgb, goal_hand_semantic, task_frame_shift)
        self._exec_command("movetoBtn", 100)

if __name__ == '__main__':
    client = RobotClient("localhost", 12345)
    import time
    print("start")
    pushbtn_goal_dof = client.get_server_value("obs/low_dim/dof_pos_obsframe")
    pushbtn_goal_base_rgb = client.get_server_value("obs/rgb/base_camera_rgb")
    pushbtn_goal_base_semantic = client.get_server_value("obs/semantic/base_camera_semantic")
    pushbtn_goal_hand_rgb = client.get_server_value("obs/rgb/hand_camera_rgb")
    pushbtn_goal_hand_semantic = client.get_server_value("obs/semantic/hand_camera_semantic")

    print("get command 0", client._get_command())

    client.cmd_moveto(np.array([0.2 , -0.1]))

    print("get command 1", client._get_command())

    goal_base_semantic = pushbtn_goal_base_semantic.get_value()[:,:,:,[1]]
    goal_hand_semantic = pushbtn_goal_hand_semantic.get_value()[:,:,:,[1]]
    client.cmd_movetoBtn(pushbtn_goal_dof, 
        pushbtn_goal_base_rgb, 
        goal_base_semantic, 
        pushbtn_goal_hand_rgb, 
        goal_hand_semantic,
        task_frame_shift = pushbtn_goal_dof
    )

    print("get command 2", client._get_command())

    import matplotlib.pyplot as plt
    plt.imshow(pushbtn_goal_base_rgb.get_value()[0])
    plt.figure()
    plt.imshow(goal_base_semantic[0,:,:,0])
    plt.figure()
    plt.imshow(pushbtn_goal_hand_rgb.get_value()[0])
    plt.figure()
    plt.imshow(goal_hand_semantic[0,:,:,0])
    plt.show()

