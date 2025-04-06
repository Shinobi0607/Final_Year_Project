const Web3 = require('web3');

// Connect to your Ethereum network (replace with your network URL)
const web3 = new Web3("http://127.0.0.1:8545"); // Ganache local instance

// Replace with your deployed contract's ABI and address
const contractABI = [ /* Paste your contract's ABI here */ ];
const contractAddress = "0xYourDeployedContractAddress";

// Create a contract instance
const contractInstance = new web3.eth.Contract(contractABI, contractAddress);

// Admin account (replace with the private key or address of your admin account)
const adminAccount = "0xYourAdminEthereumAddress";
const adminPrivateKey = "0xYourAdminPrivateKey";

// List of fixed clients
const clients = [
  { address: "0x1234567890abcdef1234567890abcdef12345678", clientID: "client_001" },
  { address: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", clientID: "client_002" },
];

// Supported version
const supportedVersion = "1.0.0";

(async () => {
  try {
    // Add clients to the contract
    for (const client of clients) {
      const addClientTx = contractInstance.methods.addClient(client.address, client.clientID);
      const gas = await addClientTx.estimateGas({ from: adminAccount });
      const data = addClientTx.encodeABI();

      const signedTx = await web3.eth.accounts.signTransaction(
        {
          to: contractAddress,
          data,
          gas,
        },
        adminPrivateKey
      );

      const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      console.log(`Client ${client.clientID} added. Tx Hash: ${receipt.transactionHash}`);
    }

    // Update the supported version
    const updateVersionTx = contractInstance.methods.updateVersion(supportedVersion);
    const gas = await updateVersionTx.estimateGas({ from: adminAccount });
    const data = updateVersionTx.encodeABI();

    const signedTx = await web3.eth.accounts.signTransaction(
      {
        to: contractAddress,
        data,
        gas,
      },
      adminPrivateKey
    );

    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    console.log(`Version updated to ${supportedVersion}. Tx Hash: ${receipt.transactionHash}`);
  } catch (error) {
    console.error("Error configuring clients:", error);
  }
})();
