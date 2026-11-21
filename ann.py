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

def forward_propogation(W,B,X):
    print("fp",X.shape,W.T.shape)
    Z = (np.dot(X,W.T)) + B
    A = sigmoid(Z)
    #print("Z",Z)
    #print("A",A)
    #print("sigmoid A",sigmoid(Z))
    return Z,np.array(A)

def layers(layer,neuron,X):
    layers = {}
    for i in range(layer):
        w,b = init(X.shape[1],neuron)
        print(f"layer {i} neuron {neuron}")
        print(f"weights {np.array(w).shape} bias {np.array(b).shape}")

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
    pprint(layers)
    return Z,A

def prediction(A):
    w, b = init(A.shape[1], 1)
    Z = np.dot(A, w.T) + b
    pred = sigmoid(Z)
    output = {"weights":w,"bias":b,"Z": Z,"previous": A,"pred": pred}
    pprint(output)
    return pred

def loss(y, pred):
    m = len(y)
    y = y.reshape(-1, 1)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    L = -np.mean(
        y * np.log(pred) +
        (1-y) * np.log(1-pred)
    )
    return L
