import numpy, math

def sigmoid(x):
    return 1/(1+numpy.exp(-x))
def sigmoidDashed(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.calcs = len(layers) - 1
        self.weights = []
        self.biases = []
        self.activations = []
        self.Z_activations = []
        self.dWeights = []
        self.dBiases = []

        for i in range(self.calcs):
            n = self.layers[i]
            k = self.layers[i+1]
            self.weights.append(
                numpy.matrix((numpy.random.rand(n*k)-0.5).reshape(k,n))
                )
            self.biases.append(
                numpy.random.rand(k)-0.5
                )
            self.dWeights.append(
                numpy.matrix(numpy.zeros(n*k).reshape(k,n))
                )
            self.dBiases.append(
                numpy.zeros(k)
                )
            self.activations.append(
                numpy.zeros(n)
                )
            self.Z_activations.append(
                numpy.zeros(n)
                )
        self.activations.append(
            numpy.zeros(self.layers[-1])
            )
        self.Z_activations.append(
            numpy.zeros(self.layers[-1])
            )

    def calculate(self, activations, layer):
        self.activations[layer] = activations
        out = self.weights[layer].dot(activations)
        out += self.biases[layer]
        out = numpy.array(out).flatten()
        self.Z_activations[layer+1] = out
        out = sigmoid(out)
        if layer < self.calcs - 1:
            return self.calculate(out, layer + 1)
        else:
            self.activations[-1] = out
            return out

    def use(self, activations):
        return self.calculate(activations, 0)

    def getCost(self, desired):
        cost = sum((self.activations[-1] - desired)**2)
        return cost

    def getResult(self):
        a = self.activations[-1].tolist()
        return a.index(max(a))

    def backPropagate(self, desired, layer, iterations):
        weights = self.weights[layer-1]
        activationsL = self.activations[layer]
        activationsL1 = self.activations[layer-1]
        biases = self.biases[layer-1]
        Z_activations = self.Z_activations[layer]

        #get change to weights
        x = 2 * (activationsL - desired) * sigmoidDashed(Z_activations)
        x = numpy.matrix(x).T
        y = numpy.matrix(activationsL1)
        dWeights = -x.dot(y)

        #get change to biases
        dBiases = -numpy.array(x).T.flatten()

        #get change to activations
        dActivations = -numpy.array(x.T.dot(weights)).flatten()

        self.dWeights[layer-1] += dWeights / iterations
        self.dBiases[layer-1] += dBiases / iterations

        desiredL1 = activationsL1 + dActivations
        if layer > 1:
            return self.backPropagate(desiredL1, layer-1, iterations)

    def train(self, data, output, iterations):
        self.dWeights = []
        self.dBiases = []
        for i in range(self.calcs):
            n = self.layers[i]
            k = self.layers[i+1]
            self.dWeights.append(
                numpy.matrix(numpy.zeros(n*k).reshape(k,n))
                )
            self.dBiases.append(
                numpy.zeros(k)
                )
        for i in range(iterations):
            self.use(data[i])
            self.backPropagate(output[i], self.calcs, iterations)
        for i in range(self.calcs):
            self.weights[i] += self.dWeights[i] * 1
            self.biases[i] += self.dBiases[i] * 1

def loadNetwork(fileName):
    def to_list(a, t=float):
        return [t(i.replace("[","").replace("]","")) for i in a.split(",")]
    file = open(fileName)
    lines = file.readlines()
    file.close()
    shape = to_list(lines[0].replace("\n",""), t=int)
    weights = []
    biases = []
    network = Network(shape)
    for i in range(len(shape) - 1):
        weights.append(numpy.matrix(to_list(lines[i+1].replace("\n","")), numpy.float64).reshape(network.weights[i].shape))
        biases.append(numpy.array(to_list(lines[i+len(shape)].replace("\n","")), numpy.float64))
    network.weights = weights
    network.biases = biases
    return network

def saveNetwork(network, fileName):
    text = ""
    text += "{}\n".format(network.layers)
    for i in network.weights + network.biases:
        l = i.tolist()
        text += "{}\n".format(l)
    file = open(fileName, "w")
    file.write(text)
    file.close()
    

#network = Network([1000,20,20,10])
#for i in range(100):
#    iterations = 10
#    data = [numpy.random.rand(1000) for i in range(iterations)]
#    output = [numpy.array([1,0,0,0,0,0,0,0,0,0],numpy.float64) for i in range(iterations)]
#    network.learn(data, output, iterations)
#d = numpy.random.rand(1000)
#a = network.use(d)
#print(a)
#saveNetwork(network, "jeff.txt")
#n = loadNetwork("jeff.txt")
#a = n.use(d)
#print(a)
