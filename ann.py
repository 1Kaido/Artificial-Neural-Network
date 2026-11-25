def init(input_neuron,neuron):
    weights = []
    bias = []
    for i in range(neuron):
      #  print("",X.type)
        W = np.random.randn(input_neuron)
        B = np.random.randn()
        weights.append(W)
        bias.append(B)
    #print(np.array(weights).shape)
    return np.array(weights),bias

def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(Z):
    return np.maximum(0, Z)

def reLU_derivative(Z):
    return (Z > 0).astype(float)

def forward_propogation(W,B,X):
    #print("fp",X.shape,W.T.shape)
    Z = (np.dot(X,W.T)) + B
    A = relu(Z)
    #print("Z",Z)
    #print("A",A)
    #print("sigmoid A",sigmoid(Z))
    return Z,np.array(A)

def build_layers(layer,neuron,X):
    layers = {}
    for i in range(layer):
        w,b = init(X.shape[1],neuron)
        #print(f"layer {i} neuron {neuron}")
        #print(f"weights {np.array(w).shape} bias {np.array(b).shape}")

        Z,A = forward_propogation(w,b,X)
        layers[i] = {
            "weights": w,
            "bias": b,
            "Z": Z,
            "A": A,
            "previous": X
        }

        pprint(f"Z {np.array(Z).shape} A {np.array(A).shape}")
        X = A
    #pprint(layers)
    return Z,A, layers

def backward_propogation(layers,output_layer,y,learning_rate=0.01):
    m = np.array(y.shape[0])
    dz = output_layer["pred"] - y 
    dw = 1/m * (dz @ output_layer["previous"])
    db = np.sum(dz)
    dw_out = output_layer["weights"]
    #update output layers also 
    #print(np.array(dw).shape,np.array(db).shape)
    #print(" h",np.array(output_layers["weights"]).shape)
    #output_layer["weights"] -= learning_rate*dw
    #output_layer["bias"] -= learning_rate*db
    #hidden layers
    for idx in reversed(layers):
        
        dA = np.dot(dz,dw_out.T)
        dZ = dA * reLU_derivative(layers[idx]["Z"])
        dW = dZ.T @ layers[idx]["previous"] / m 
        #print(np.array(dZ).shape)
        dB = 1/m * np.sum(dZ)

        #update 
        #print(np.array(dW).shape,np.array(dB).shape)
        #print(" h",np.array(layers[idx]["weights"]).shape)
        layers[idx]["weights"] = layers[idx]["weights"] - learning_rate*dW
        layers[idx]["bias"] = layers[idx]["bias"] - learning_rate*dB

        #now pass new val to next layer 
        dz = dZ
        dw = dw_out

def prediction(A):
    w, b = init(A.shape[1], 1)
    Z = np.dot(A, w.T) + b
    pred = sigmoid(Z)
    #print("prediction output")
    output = {"weights":w,"bias":b,"Z": Z,"previous": A,"pred": pred}
    #pprint(output)
    return pred,output