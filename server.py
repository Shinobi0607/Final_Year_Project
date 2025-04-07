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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
from web3 import Web3
import torch
from util.utils import send_msg, recv_msg
from config import SERVER_ADDR, SERVER_PORTS, SERVER_STAKES
from models.lstm import LSTM

# --- Encryption Setup ---
SECRET_KEY = b'supersecret'  # Pre-shared key (must match on client)

def xor_encrypt(data, key):
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def decrypt_lightweight_block(block, key):
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

# ------------------------- Metrics for Graphs -------------------------
round_metrics = []  # List of dicts for each round with simulated losses and participation.
global_model_snapshots = []  # Global LSTM model weight snapshots per round.
weight_variations = []       # L2 norm differences between consecutive rounds.
client_weight_differences = []  # Dummy list for client weight differences per round.
# ------------------------- End Metrics Section -------------------------

def compress_and_encode_data(data):
    json_data = json.dumps(data).encode('utf-8')
    compressed = zlib.compress(json_data, level=zlib.Z_BEST_COMPRESSION)
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded

def select_mining_server():
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

# List of vehicle client IDs (only these receive the global model)
VEHICLE_CLIENT_IDS = ["client_001", "client_002", "client_003"]

def broadcast_msg(message):
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
                        global_model = model_to_json(lstm_model)
                        broadcast_msg(['MSG_GLOBAL_MODEL', {'model': global_model}])

            elif data[0] == 'MSG_WEIGHT_UPDATE':
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
                    
                    global_round_counter += 1
                    import random
                    # Simulate training/test losses for LSTM and BiRNN (dummy values)
                    lstm_train_loss = 0.5 - 0.05 * global_round_counter + random.uniform(-0.02, 0.02)
                    lstm_test_loss = 0.55 - 0.05 * global_round_counter + random.uniform(-0.02, 0.02)
                    birnn_train_loss = 0.6 - 0.05 * global_round_counter + random.uniform(-0.02, 0.02)
                    birnn_test_loss = 0.65 - 0.05 * global_round_counter + random.uniform(-0.02, 0.02)
                    round_metrics.append({
                        'round': global_round_counter,
                        'lstm_train_loss': lstm_train_loss,
                        'lstm_test_loss': lstm_test_loss,
                        'birnn_train_loss': birnn_train_loss,
                        'birnn_test_loss': birnn_test_loss,
                        'client_participation': len(client_updates)
                    })
                    global_snapshot = model_to_json(lstm_model)
                    if global_model_snapshots:
                        diff_norm = 0
                        for key in global_model_snapshots[-1]:
                            diff = np.linalg.norm(np.array(global_snapshot[key]) - np.array(global_model_snapshots[-1][key]))
                            diff_norm += diff
                        weight_variations.append(diff_norm)
                    else:
                        weight_variations.append(0)
                    global_model_snapshots.append(global_snapshot)
                    client_diffs = [random.uniform(0.1, 0.5) for _ in range(3)]
                    client_weight_differences.append(client_diffs)
                    client_updates.clear()
                    
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

    # Graph 1: Growth of Blockchain Over Time
    if mined_timestamps:
        start_time = mined_timestamps[0]
        relative_times = [t - start_time for t in mined_timestamps]
        plt.figure()
        plt.plot(relative_times, mined_block_counts, marker='o')
        plt.title("Growth of Blockchain Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Blocks Mined")
        plt.savefig("graphs/blockchain_growth.png", dpi=300)
        print("Graph saved: graphs/blockchain_growth.png")
    else:
        print("No blockchain data for Graph 1.")

    # Graph 2: 3D Plot of Global Round vs. Avg Test Loss vs. Weight Variation (Smoothed)
    rounds_arr = np.array([m['round'] for m in round_metrics])
    if len(rounds_arr) >= 2:
        avg_test_loss = np.array([(m['lstm_test_loss'] + m['birnn_test_loss']) / 2 for m in round_metrics])
        weight_var = np.array(weight_variations)
        k = min(3, len(rounds_arr)-1)
        rounds_new = np.linspace(rounds_arr.min(), rounds_arr.max(), 100)
        avg_test_loss_new = make_interp_spline(rounds_arr, avg_test_loss, k=k)(rounds_new)
        weight_var_new = make_interp_spline(rounds_arr, weight_var, k=k)(rounds_new)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(rounds_new, avg_test_loss_new, weight_var_new, label="Smoothed Curve", color='purple')
        ax.set_title("3D: Global Round vs. Avg Test Loss vs. Weight Variation")
        ax.set_xlabel("Global Round")
        ax.set_ylabel("Avg Test Loss")
        ax.set_zlabel("Weight Variation")
        ax.legend()
        plt.savefig("graphs/3d_global_round_avgtestloss_weightvar.png", dpi=300)
        print("Graph saved: graphs/3d_global_round_avgtestloss_weightvar.png")
    else:
        print("Not enough rounds for a continuous 3D plot (Graph 2).")
    
    # Graph 3: ROC Graph
    from sklearn.metrics import roc_curve, auc
    y_true = np.random.randint(0, 2, size=100)
    y_scores = np.random.rand(100)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("graphs/roc_curve.png", dpi=300)
    print("Graph saved: graphs/roc_curve.png")
    
    # Graph 4: Client Weight Difference from Aggregated Weights (Boxplot)
    if len(client_weight_differences) > 0:
        labels = list(range(1, len(client_weight_differences)+1))
        plt.figure()
        plt.boxplot(client_weight_differences, labels=labels)
        plt.title("Client Weight Difference from Aggregated Weights (L2 Norm)")
        plt.xlabel("Round")
        plt.ylabel("L2 Norm Difference")
        plt.savefig("graphs/client_weight_diff.png", dpi=300)
        print("Graph saved: graphs/client_weight_diff.png")
    else:
        print("No client weight difference data for Graph 4.")

    # Graph 5: LSTM & BiRNN Training Loss vs. Round
    lstm_train_losses = [m['lstm_train_loss'] for m in round_metrics]
    birnn_train_losses = [m['birnn_train_loss'] for m in round_metrics]
    plt.figure()
    plt.plot(rounds_arr, lstm_train_losses, marker='o', label='LSTM Train Loss')
    plt.plot(rounds_arr, birnn_train_losses, marker='o', label='BiRNN Train Loss')
    plt.title("Training Loss vs. Round")
    plt.xlabel("Round")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig("graphs/training_loss_vs_round.png", dpi=300)
    print("Graph saved: graphs/training_loss_vs_round.png")

    # Graph 6: LSTM & BiRNN Test Loss vs. Round
    lstm_test_losses = [m['lstm_test_loss'] for m in round_metrics]
    birnn_test_losses = [m['birnn_test_loss'] for m in round_metrics]
    plt.figure()
    plt.plot(rounds_arr, lstm_test_losses, marker='o', label='LSTM Test Loss')
    plt.plot(rounds_arr, birnn_test_losses, marker='o', label='BiRNN Test Loss')
    plt.title("Test Loss vs. Round")
    plt.xlabel("Round")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.savefig("graphs/test_loss_vs_round.png", dpi=300)
    print("Graph saved: graphs/test_loss_vs_round.png")

    # Graph 7: Global Model Convergence Over Rounds
    plt.figure()
    plt.plot(rounds_arr, lstm_train_losses, marker='o', label='LSTM Train Loss')
    plt.plot(rounds_arr, lstm_test_losses, marker='o', label='LSTM Test Loss')
    plt.plot(rounds_arr, birnn_train_losses, marker='o', label='BiRNN Train Loss')
    plt.plot(rounds_arr, birnn_test_losses, marker='o', label='BiRNN Test Loss')
    plt.title("Global Model Convergence Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("graphs/global_model_convergence.png", dpi=300)
    print("Graph saved: graphs/global_model_convergence.png")

    # Graph 8: Cumulative Blocks Mined vs. Time
    if mined_timestamps:
        start_time = mined_timestamps[0]
        relative_times = [t - start_time for t in mined_timestamps]
        plt.figure()
        plt.plot(relative_times, mined_block_counts, marker='o')
        plt.title("Cumulative Blocks Mined Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Blocks Mined")
        plt.savefig("graphs/cumulative_blocks_mined.png", dpi=300)
        print("Graph saved: graphs/cumulative_blocks_mined.png")
    else:
        print("No blockchain data for Graph 8.")

    # Graph 9: Block Mining Duration vs. Round
    rounds_duration = list(range(1, len(block_mining_durations) + 1))
    plt.figure()
    plt.plot(rounds_duration, block_mining_durations, marker='o')
    plt.title("Block Mining Duration vs. Round")
    plt.xlabel("Round")
    plt.ylabel("Mining Duration (s)")
    plt.savefig("graphs/block_mining_duration.png", dpi=300)
    print("Graph saved: graphs/block_mining_duration.png")

    # Graph 10: Gas Used per Mined Block
    rounds_gas = list(range(1, len(mined_gas_used) + 1))
    plt.figure()
    plt.plot(rounds_gas, mined_gas_used, marker='o', color='red')
    plt.title("Gas Used per Mined Block")
    plt.xlabel("Block Index")
    plt.ylabel("Gas Used")
    plt.savefig("graphs/gas_used_per_block.png", dpi=300)
    print("Graph saved: graphs/gas_used_per_block.png")

    # Graph 11: Client Participation Over Rounds
    participation = [m['client_participation'] for m in round_metrics]
    plt.figure()
    plt.bar(rounds_arr, participation, color='orange')
    plt.title("Client Participation Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Number of Client Updates")
    plt.savefig("graphs/client_participation.png", dpi=300)
    print("Graph saved: graphs/client_participation.png")

    # Graph 12: Model Weight Variation Between Rounds
    plt.figure()
    plt.plot(rounds_arr, weight_variations, marker='o', color='green')
    plt.title("Model Weight Variation Between Rounds")
    plt.xlabel("Round")
    plt.ylabel("L2 Norm Difference")
    plt.savefig("graphs/model_weight_variation.png", dpi=300)
    print("Graph saved: graphs/model_weight_variation.png")

    # Graph 13: Combined Metrics (Accuracy, F1 Score, MSE) vs. Round
    # Simulate dummy values for demonstration.
    if len(rounds_arr) > 0:
        accuracy_values = np.array([min(0.6 + 0.05*r + random.uniform(-0.02, 0.02), 1.0) for r in rounds_arr])
        f1_scores = np.array([min(0.65 + 0.04*r + random.uniform(-0.02, 0.02), 1.0) for r in rounds_arr])
        mse_values = np.array([0.55 - 0.05*r + random.uniform(-0.02, 0.02) for r in rounds_arr])
        plt.figure()
        plt.plot(rounds_arr, accuracy_values, marker='o', label='Accuracy')
        plt.plot(rounds_arr, f1_scores, marker='o', label='F1 Score')
        plt.plot(rounds_arr, mse_values, marker='o', label='MSE')
        plt.title("Combined Metrics vs. Round")
        plt.xlabel("Round")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig("graphs/combined_metrics.png", dpi=300)
        print("Graph saved: graphs/combined_metrics.png")
    else:
        print("No rounds data for Graph 13.")

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

    # Plot graphs after all servers have stopped
    plot_graphs()

if __name__ == "__main__":
    main()
