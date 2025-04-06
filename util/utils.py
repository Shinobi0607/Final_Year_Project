import numpy as np
import pickle
import struct
import socket
import math
import time
import os

def send_msg(sock, msg, timeout=10):
    """
    Sends a serialized message over the given socket.
    :param sock: The socket to send the message over.
    :param msg: The message to be sent.
    :param timeout: The timeout for sending the message.
    """
    try:
        orig_timeout = sock.gettimeout()
        sock.settimeout(timeout)
        
        # Serialize the message
        msg_pickle = pickle.dumps(msg)
        
        # Send the length of the serialized message, followed by the message itself
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)
        
        print(f"[SEND] {msg[0]} sent to {sock.getpeername()}")

        # Reset socket timeout
        sock.settimeout(orig_timeout)

    except socket.timeout:
        print(f"[ERROR] Sending message timed out to {sock.getpeername()}")
        raise
    except pickle.PickleError as e:
        print(f"[ERROR] Failed to serialize message: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to send message: {e}")
        raise

def recv_msg(sock, expect_msg_type=None, timeout=10):
    """
    Receives a serialized message from the given socket.
    :param sock: The socket to receive the message from.
    :param expect_msg_type: The expected message type.
    :param timeout: The timeout for receiving the message.
    :return: The deserialized message or None if an error occurs.
    """
    try:
        orig_timeout = sock.gettimeout()
        sock.settimeout(timeout)

        # Receive the length of the message
        msg_len_data = sock.recv(4)
        if len(msg_len_data) < 4:
            print("[WARNING] Received incomplete length data. Connection may be closed.")
            return None

        msg_len = struct.unpack(">I", msg_len_data)[0]

        # Receive the full message data
        msg_data = b''
        while len(msg_data) < msg_len:
            chunk = sock.recv(min(4096, msg_len - len(msg_data)))
            if not chunk:
                print("[WARNING] Connection closed before full message received.")
                return None
            msg_data += chunk

        # Deserialize the message
        msg = pickle.loads(msg_data)
        print(f"[RECV] {msg[0]} received from {sock.getpeername()}")

        # Verify the message type, if specified
        if expect_msg_type is not None and msg[0] != expect_msg_type:
            raise ValueError(f"Unexpected message type. Expected {expect_msg_type}, got {msg[0]}")

        # Reset socket timeout
        sock.settimeout(orig_timeout)

        return msg

    except socket.timeout:
        print(f"[ERROR] Receiving message timed out from {sock.getpeername()}")
        return None
    except (pickle.PickleError, ValueError) as e:
        print(f"[ERROR] Message deserialization error: {e}")
        return None
    except ConnectionError as e:
        print(f"[ERROR] Connection error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error in recv_msg: {e}")
        return None

def moving_average(param_mvavr, param_new, movingAverageHoldingParam):
    """
    Computes the moving average of a parameter.
    :param param_mvavr: The previous moving average value.
    :param param_new: The new value to include in the moving average.
    :param movingAverageHoldingParam: The weight for the previous average in the calculation.
    :return: The updated moving average.
    """
    if param_mvavr is None or np.isnan(param_mvavr):
        param_mvavr = param_new
    else:
        if not np.isnan(param_new):
            param_mvavr = movingAverageHoldingParam * param_mvavr + (1 - movingAverageHoldingParam) * param_new
    return param_mvavr

def log_message(log_file, message):
    """
    Logs a message to the specified log file.
    :param log_file: The file path where the message should be logged.
    :param message: The message to log.
    """
    try:
        with open(log_file, "a") as log:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log.write(f"[{timestamp}] {message}\n")
    except IOError as e:
        print(f"[ERROR] Failed to write to log file {log_file}: {e}")

