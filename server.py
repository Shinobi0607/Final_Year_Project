import random
import socket
import threading
import time
import json
import zlib
import base64
import hashlib
import matplotlib.pyplot as plt
import os
from web3 import Web3
import torch
from util.utils import send_msg, recv_msg
from config import SERVER_ADDR, SERVER_PORTS, SERVER_STAKES
from models.lstm import LSTM

# --- Encryption Setup ---
SECRET_KEY = b'supersecret'  # Pre-shared key (must match on client)

def xor_encrypt(data, key):
    # Simple XOR encryption (symmetric)
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def decrypt_lightweight_block(block, key):
    # Decrypt a lightweight block: base64 decode -> XOR decrypt -> decompress -> JSON decode
    encrypted = base64.b64decode(block)
    decrypted_compressed = xor_encrypt(encrypted, key)
    decompressed = zlib.decompress(decrypted_compressed)
    data = json.loads(decompressed.decode('utf-8'))
    return data

# --- Blockchain and Model Setup ---
blockchain_url = "http://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(blockchain_url))
if web3.is_connected():
    print("Connected to blockchain.")
else:
    raise Exception("Failed to connect to the blockchain.")

with open("abi.json") as f:
    contract_data = json.load(f)
    contract_abi = contract_data["abi"]

contract_address = Web3.to_checksum_address("0x5FbDB2315678afecb367f032d93F642f64180aa3")
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Dictionaries and locks
server_clients = {port: [] for port in SERVER_PORTS}
client_updates = {}
client_lock = threading.Lock()
active_clients = {}  # key: client_id, value: {address, server_port, last_active, socket}

# Global variables for blockchain logging
mined_timestamps = []
mined_block_counts = []
mined_gas_used = []
block_counter = 0

block_creation_times = []
block_mining_durations = []

# We require 3 rounds (and 3 blocks mined) to then stop the system.
MAX_ROUNDS = 3
global_round_counter = 0

stop_event = threading.Event()
initial_server_start_time = None

# Global dictionary for RSU defect alerts.
rsu_defect_reports = {}  # key: rsu_id, value: set of client_ids reporting defect
expected_reports = 3     # Wait for 3 unique client reports

# To enforce that each block is mined by a different server,
# we keep track of mining servers used.
mining_servers_used = set()

def compress_and_encode_data(data):
    json_data = json.dumps(data).encode('utf-8')
    compressed = zlib.compress(json_data, level=zlib.Z_BEST_COMPRESSION)
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded

def select_mining_server():
    # Choose only from servers not used in previous rounds.
    available_servers = {server: stake for server, stake in SERVER_STAKES.items() if server not in mining_servers_used}
    if not available_servers:
        available_servers = SERVER_STAKES  # fallback if needed
    total_stake = sum(available_servers.values())
    chosen_stake = random.uniform(0, total_stake)
    cumulative_stake = 0
    for server, stake in available_servers.items():
        cumulative_stake += stake
        if chosen_stake <= cumulative_stake:
            mining_servers_used.add(server)
            return server
    server = list(available_servers.keys())[0]
    mining_servers_used.add(server)
    return server

def mine_block(serialized_updates):
    global block_counter
    try:
        selected_server = select_mining_server()
        print(f"Selected server for mining: {selected_server}")
        sender = web3.eth.accounts[0]

        encoded_data = compress_and_encode_data(serialized_updates)
        data_hash = hashlib.sha256(encoded_data.encode('utf-8')).hexdigest()

        start_time = time.time()
        creation_time = start_time - initial_server_start_time
        tx_hash = contract.functions.mineBlock(data_hash).transact({
            'from': sender,
            'gas': 8000000
        })
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        end_time = time.time()

        block_counter += 1
        mined_timestamps.append(end_time)
        mined_block_counts.append(block_counter)
        mined_gas_used.append(receipt.gasUsed)

        mining_duration = end_time - start_time
        block_creation_times.append(creation_time)
        block_mining_durations.append(mining_duration)

        print(f"Block mined successfully by {selected_server} with transaction hash: {tx_hash.hex()}")

        return receipt
    except Exception as e:
        print(f"Error mining block: {e}")
        return None

def fedprox_aggregate_weights(model_updates, server_model, mu=0.01):
    aggregated_model = {key: torch.zeros_like(param) for key, param in server_model.state_dict().items()}
    total_clients = 0
    for client_idx, weights in enumerate(model_updates):
        try:
            for key, server_param in server_model.state_dict().items():
                if key in weights:
                    client_param = torch.tensor(weights[key]).clone().detach()
                    if client_param.shape != server_param.shape:
                        print(f"Warning: Shape mismatch for {key}. Skipping client {client_idx}.")
                        raise ValueError(f"Shape mismatch for key {key}")
                    prox_term = mu * (client_param - server_param)
                    aggregated_model[key] += (client_param - prox_term)
                else:
                    print(f"Warning: Missing key {key}. Using server model's values.")
                    aggregated_model[key] += server_param
            total_clients += 1
        except Exception as e:
            print(f"Skipping client {client_idx} due to error: {e}")
    if total_clients == 0:
        raise ValueError("No valid updates to aggregate.")
    for key in aggregated_model.keys():
        aggregated_model[key] /= total_clients
    return aggregated_model

def verify_rsu_defect(payload):
    print("Verifying RSU defect alert with payload:", payload)
    return True

def model_to_json(model):
    state = model.state_dict()
    json_compatible = {}
    for key, tensor in state.items():
        json_compatible[key] = tensor.cpu().detach().numpy().tolist()
    return json_compatible

# List of vehicle client IDs (to broadcast global model only to vehicles)
VEHICLE_CLIENT_IDS = ["client_001", "client_002", "client_003"]

def broadcast_msg(message):
    # Short delay to help slow clients catch up
    time.sleep(2)
    with client_lock:
        print(f"[BROADCAST] Active clients at broadcast: {list(active_clients.keys())}")
        for cid, info in active_clients.items():
            if cid not in VEHICLE_CLIENT_IDS:
                continue
            try:
                client_sock = info.get('socket')
                if client_sock:
                    send_msg(client_sock, message)
                    print(f"[SEND] MSG_GLOBAL_MODEL sent to {cid}")
            except Exception as e:
                print(f"[ERROR] Could not send model to {cid}: {e}")

def handle_client(sock, addr, port, lstm_model):
    global global_round_counter
    client_id = None
    try:
        msg = recv_msg(sock)
        if not msg or msg[0] != 'MSG_CLIENT_DATA':
            print("[ERROR] Did not receive valid client data.")
            return

        data = msg[1]
        client_id = data['client_id']
        with client_lock:
            if client_id in active_clients:
                print(f"[DUPLICATE] Client {client_id} already connected.")
                return
            active_clients[client_id] = {
                'address': addr,
                'server_port': port,
                'last_active': time.time(),
                'socket': sock
            }
            print(f"[CONNECTED] Client {client_id} connected from {addr}. Active clients: {list(active_clients.keys())}")

        send_msg(sock, ['MSG_INIT_SERVER_TO_CLIENT', 'LSTM', lstm_model.hidden_dim, 0.001, 32, 5])

        while not stop_event.is_set():
            data = recv_msg(sock)
            if data is None:
                time.sleep(0.5)
                continue

            with client_lock:
                active_clients[client_id]['last_active'] = time.time()

            if data[0] == 'MSG_RSU_DEFECT_ALERT':
                payload = data[1]
                rsu_id = payload['rsu_id']
                print(f"[ALERT] {client_id} reports RSU defect: {rsu_id}")
                with client_lock:
                    rsu_defect_reports.setdefault(rsu_id, set()).add(client_id)
                    print(f"Current reports for {rsu_id}: {rsu_defect_reports[rsu_id]}")
                if len(rsu_defect_reports[rsu_id]) == expected_reports:
                    if verify_rsu_defect(payload):
                        print(f"[FIRMWARE] Sending update to RSU {rsu_id}")
                        rsu_info = active_clients.get(rsu_id)
                        if rsu_info:
                            send_msg(rsu_info['socket'], ['MSG_FIRMWARE_DEPLOY', {'status': 'firmware_update'}])
                        else:
                            print(f"RSU {rsu_id} not connected.")
                        # Distribute model to all vehicle clients
                        global_model = model_to_json(lstm_model)
                        broadcast_msg(['MSG_GLOBAL_MODEL', {'model': global_model}])

            elif data[0] == 'MSG_WEIGHT_UPDATE':
                # Decrypt the lightweight block received from the client
                encrypted_block = data[1]["weights"]
                try:
                    weights = decrypt_lightweight_block(encrypted_block, SECRET_KEY)
                except Exception as e:
                    print(f"[ERROR] Failed to decrypt block from {client_id}: {e}")
                    continue
                print(f"[MODEL] Received lightweight block from {client_id} at {time.time()}")
                with client_lock:
                    client_updates[client_id] = weights
                if len(client_updates) == 3:
                    aggregated = fedprox_aggregate_weights(
                        [{k: torch.tensor(v) for k, v in update.items()} for update in client_updates.values()],
                        lstm_model
                    )
                    lstm_model.load_state_dict(aggregated)
                    mine_block(client_updates)
                    broadcast_msg(['MSG_GLOBAL_MODEL', {'model': model_to_json(lstm_model)}])
                    client_updates.clear()
                    global_round_counter += 1
                    print(f"[ROUND] Completed round {global_round_counter} of global updates.")
                    if global_round_counter >= MAX_ROUNDS:
                        print("[INFO] Reached maximum rounds. Stopping server.")
                        stop_event.set()
    except Exception as e:
        print(f"[ERROR] Exception handling client {client_id}: {e}")
    finally:
        if client_id:
            with client_lock:
                active_clients.pop(client_id, None)
                print(f"[DISCONNECTED] Client {client_id} disconnected. Active clients: {list(active_clients.keys())}")
        sock.close()

def start_server(server_port, lstm_model):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_ADDR, server_port))
    server_socket.listen(5)
    print(f"Server started on {SERVER_ADDR}:{server_port}")

    while not stop_event.is_set():
        server_socket.settimeout(1.0)
        try:
            client_sock, addr = server_socket.accept()
            threading.Thread(target=handle_client, args=(client_sock, addr, server_port, lstm_model)).start()
        except socket.timeout:
            pass

    server_socket.close()
    print(f"Server on {SERVER_ADDR}:{server_port} stopped.")

def plot_graphs():
    os.makedirs("graphs", exist_ok=True)

    if not mined_timestamps:
        print("No blocks mined, no data to plot.")
        return

    start_time = mined_timestamps[0]
    relative_times = [t - start_time for t in mined_timestamps]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(relative_times, mined_block_counts, marker='o')
    plt.title("Number of Mined Blocks Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Blocks Mined")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(mined_gas_used) + 1), mined_gas_used, marker='x', color='red')
    plt.title("Gas Used per Mined Block")
    plt.xlabel("Block Index")
    plt.ylabel("Gas Used")

    plt.tight_layout()
    plt.savefig("graphs/blockchain_plots.png", dpi=300)
    print("Plots saved to 'graphs/blockchain_plots.png'")

    plt.figure(figsize=(6, 4))
    plt.scatter(block_creation_times, block_mining_durations, color='green', marker='o')
    plt.title("Block Creation Time vs Block Mining Time")
    plt.xlabel("Block Creation Time (s since server start)")
    plt.ylabel("Block Mining Duration (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/block_creation_vs_mining_time.png", dpi=300)
    print("Plots saved to 'graphs/block_creation_vs_mining_time.png'")

def main():
    global initial_server_start_time
    initial_server_start_time = time.time()

    # Initialize LSTM model parameters
    input_dim = 3  
    hidden_dim = 100
    output_dim = 1
    lstm_model = LSTM(input_dim, hidden_dim, output_dim)

    threads = []
    for port in SERVER_PORTS:
        thread = threading.Thread(target=start_server, args=(port, lstm_model))
        thread.start()
        threads.append(thread)

    stop_event.wait()

    for thread in threads:
        thread.join()

    plot_graphs()

if __name__ == "__main__":
    main()
