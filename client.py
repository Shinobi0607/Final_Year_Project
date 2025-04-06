import time, random, argparse, socket, json, zlib, base64
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models.lstm import LSTM
from models.bi_rnn import BiRNN
from util.utils import send_msg, recv_msg
from config import SERVER_ADDR, SERVER_PORTS

# --- Encryption Setup ---
SECRET_KEY = b'supersecret'

def xor_encrypt(data, key):
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def create_lightweight_block(data, key):
    json_data = json.dumps(data).encode('utf-8')
    compressed = zlib.compress(json_data, level=zlib.Z_BEST_COMPRESSION)
    encrypted = xor_encrypt(compressed, key)
    encoded = base64.b64encode(encrypted).decode('utf-8')
    return encoded

CLIENT_ETH_ADDRESSES = {
    "client_001": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
    "client_002": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
    "client_003": "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
}

VEHICLE_CLIENTS = ["client_001", "client_002", "client_003"]
RSU_CLIENTS = ["client_004"]
CLIENT_VEHICLE_MAPPING = {
    "client_001": "veh_001",
    "client_002": "veh_002",
    "client_003": "veh_003",
    "client_004": "veh_004"
}

NUM_EPOCHS = 5
LEARNING_RATE = 0.001

def setup_socket():
    port = random.choice(SERVER_PORTS)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_ADDR, port))
        print(f"[INFO] Connected to server on port {port}")
        return sock
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return None

def state_dict_to_json(sd):
    return {k: v.cpu().detach().numpy().tolist() for k, v in sd.items()}

def send_weight_update(sock, client_id, model_type, model_weights):
    try:
        block = create_lightweight_block(model_weights, SECRET_KEY)
        send_msg(sock, ['MSG_WEIGHT_UPDATE', {
            'client_id': client_id,
            'model_type': model_type,
            'weights': block
        }])
        print(f"[SEND] {model_type} weights sent from {client_id}")
    except Exception as e:
        print(f"[ERROR] Sending {model_type} weights failed: {e}")

def perform_firmware_update():
    print("[FIRMWARE] Simulating firmware update...")
    time.sleep(2)
    print("[FIRMWARE] Update completed.")

def load_local_dataset(vehicle_id, csv_path="vanet_data.csv"):
    try:
        df = pd.read_csv(csv_path)
        df = df[df['vehicle_id'] == vehicle_id]
        print(f"[DATA] Loaded {len(df)} rows for {vehicle_id}")
        return df
    except Exception as e:
        print(f"[ERROR] Loading data failed: {e}")
        return None

def preprocess_data(df):
    try:
        df = df.dropna(subset=['lat', 'lon', 'reputation_value', 'defected'])
        features = df[['lat', 'lon', 'reputation_value']].values.astype(np.float32)
        targets = df['defected'].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        return (
            torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1),
            torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1)
        )
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None, None, None, None

def train_and_evaluate(model, model_name, train_X, train_y, test_X, test_y):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.unsqueeze(0)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{model_name.upper()}] Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(test_X.unsqueeze(0))
        test_loss = criterion(predictions, test_y).item()
        print(f"[{model_name.upper()}] Test Loss on 20%: {test_loss:.4f}")

    return model.state_dict()

def main():
    all_clients = VEHICLE_CLIENTS + RSU_CLIENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', required=True, choices=all_clients)
    args = parser.parse_args()

    client_id = args.client_id
    is_vehicle = client_id in VEHICLE_CLIENTS
    is_rsu = client_id in RSU_CLIENTS

    eth_address = CLIENT_ETH_ADDRESSES.get(client_id, "")
    vehicle_id = CLIENT_VEHICLE_MAPPING[client_id]

    sock = setup_socket()
    if not sock:
        return

    try:
        send_msg(sock, ['MSG_CLIENT_DATA', {
            'client_id': client_id,
            'client_version': "2.0.0",
            'eth_address': eth_address
        }])
        print(f"[SEND] MSG_CLIENT_DATA sent to {sock.getpeername()}")

        init_msg = recv_msg(sock)
        if not init_msg or init_msg[0] != 'MSG_INIT_SERVER_TO_CLIENT':
            print("[ERROR] Initialization failed.")
            return

        print(f"[RECV] MSG_INIT_SERVER_TO_CLIENT received from {sock.getpeername()}")

        if is_vehicle:
            send_msg(sock, ['MSG_RSU_DEFECT_ALERT', {
                'rsu_id': 'client_004',
                'defect_info': 'signal_failure'
            }])
            print(f"[ALERT] {client_id} sent defect alert about RSU client_004")

        timeout_seconds = 100
        last_message_time = time.time()

        while True:
            now = time.time()
            if now - last_message_time > timeout_seconds:
                print(f"[TIMEOUT] No message received for {timeout_seconds} seconds. Closing connection.")
                break

            data = recv_msg(sock)
            if data is None:
                time.sleep(1)
                continue

            last_message_time = time.time()
            msg_type = data[0]
            payload = data[1] if len(data) > 1 else {}

            if msg_type == 'MSG_FIRMWARE_DEPLOY':
                if is_rsu:
                    perform_firmware_update()
                else:
                    print("[INFO] Vehicle received firmware message. Ignoring.")

            elif msg_type == 'MSG_GLOBAL_MODEL':
                print(f"[MODEL] {client_id} received global model.")

                df = load_local_dataset(vehicle_id)
                train_X, train_y, test_X, test_y = preprocess_data(df) if df is not None else (None, None, None, None)
                if train_X is None:
                    continue

                # --- Train LSTM ---
                lstm_model = LSTM(input_dim=3, hidden_dim=100, output_dim=1)
                lstm_model.load_state_dict({k: torch.tensor(v) for k, v in payload['model'].items()})
                lstm_weights = train_and_evaluate(lstm_model, "lstm", train_X, train_y, test_X, test_y)
                send_weight_update(sock, client_id, "lstm", state_dict_to_json(lstm_weights))

                # --- Train BiRNN ---
                birnn_model = BiRNN(input_dim=3, hidden_dim=100, output_dim=1)
                birnn_model.load_state_dict({k: torch.tensor(v) for k, v in payload['model'].items()})
                birnn_weights = train_and_evaluate(birnn_model, "birnn", train_X, train_y, test_X, test_y)
                send_weight_update(sock, client_id, "birnn", state_dict_to_json(birnn_weights))

            else:
                print(f"[INFO] Unknown message type: {msg_type}")

    except KeyboardInterrupt:
        print("[INFO] Client manually interrupted. Shutting down.")

    finally:
        sock.close()
        print("[DISCONNECT] Client disconnected.")

if __name__ == "__main__":
    main()
