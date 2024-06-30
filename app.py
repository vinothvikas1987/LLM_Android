from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model at the start of the application
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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
    return {"status": "success", "details": transaction_details, "date": formatted_date}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
