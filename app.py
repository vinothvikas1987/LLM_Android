from flask import Flask, request, jsonify
from transformers import AutoModel, AutoConfig
import torch

app = Flask(__name__)

# Load the model at the start of the application
config = AutoConfig.from_pretrained('model/config.json')
model = AutoModel.from_pretrained('model/model.safetensors', config=config)

@app.route('/endpoint', methods=['POST'])
def endpoint():
    data = request.get_json()
    transaction_details = data.get('transactionDetails')
    formatted_date = data.get('formattedDate')

    # Print the received data to the terminal
    print("Received data:")
    print("Transaction Details:", transaction_details)
    print("Formatted Date:", formatted_date)

    # Perform inference using the model
    result = model_inference(transaction_details, formatted_date)

    return jsonify({"result": result})

def model_inference(transaction_details, formatted_date):
    # Your model inference logic here
    # This is a placeholder for actual inference logic
    # Use the model to generate predictions
    inputs = {'input_data': transaction_details}  # Adjust this line based on your input format
    with torch.no_grad():
        outputs = model(**inputs)
    # Process the outputs as needed
    return {"status": "success", "details": transaction_details, "date": formatted_date}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
