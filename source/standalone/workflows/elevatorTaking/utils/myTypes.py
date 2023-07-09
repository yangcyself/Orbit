# A simple implmentation of a type system for python. Mainly for passing numpy arrays via TCP.
import numpy as np
import struct

class myNumpyArray:
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data

    def to_bytes(self):
        value = self.data
        shape = value.shape
        dtype_str = str(value.dtype)
        packed_data = struct.pack('!II', len(shape), len(dtype_str))
        for dim in shape:
            packed_data += struct.pack('!I', dim)
        packed_data += dtype_str.encode("utf-8")
        packed_data += value.tobytes()
        return packed_data

    def bytes_length(self):
        return len(self.to_bytes())

    @staticmethod
    def from_buffer(buffer):
        shape = []
        dtype_str = ''
        offset = 0
        shape_len, dtype_len = struct.unpack('!II', buffer[offset:offset+8])
        offset += 8
        for _ in range(shape_len):
            dim, = struct.unpack('!I', buffer[offset:offset+4])
            offset += 4
            shape.append(dim)
        dtype_str = buffer[offset:offset+dtype_len].decode("utf-8")
        offset += dtype_len
        dtype = np.dtype(dtype_str)

        value = np.frombuffer(buffer, 
            count = np.prod(shape),
            dtype=dtype, 
            offset=offset).reshape(shape)
        offset += value.nbytes
        return value, offset

class myStr:
    def __init__(self, data):
        assert isinstance(data, str)
        self.data = data
    
    def to_bytes(self):
        value = self.data
        packed_data = struct.pack('!I', len(value))
        packed_data += value.encode("utf-8")
        return packed_data

    def bytes_length(self):
        return 4 + len(self.data.encode("utf-8"))

    @staticmethod
    def from_buffer(buffer):
        length, = struct.unpack('!I', buffer[:4])
        value = buffer[4:4+length].decode("utf-8")
        return value, 4 + length

def baseTypeFactory(fmt):
    class myBaseType:
        def __init__(self, data):
            self.data = data

        def to_bytes(self):
            return struct.pack(fmt, self.data)

        def bytes_length(self):
            return struct.calcsize(fmt)

        @staticmethod
        def from_buffer(buffer):
            value, = struct.unpack(fmt, buffer)
            return value, struct.calcsize(fmt)
    return myBaseType

myInt = baseTypeFactory('!i')
myFloat = baseTypeFactory('!f')

