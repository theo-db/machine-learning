import numpy, math, threading

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

    def backPropagate(self, activations, desired, layer, iterations, acts=None, z_acts=None):
        #calculate layer activations
        if activations is not None:
            acts = []
            z_acts = []
            for i in self.layers:
                acts.append(numpy.zeros(i, numpy.float64))
                z_acts.append(numpy.zeros(i, numpy.float64))
            def c(a,l):
                acts[l] = a
                o = self.weights[l].dot(a)
                o += self.biases[l]
                o = numpy.array(o).flatten()
                z_acts[l+1] = o
                o = sigmoid(o)
                if l < self.calcs - 1: return c(o, l+1)
                else:
                    acts[-1] = o
                    return o
            c(activations, 0)        
        
        weights = self.weights[layer-1]
        activationsL = acts[layer]
        activationsL1 = acts[layer-1]
        biases = self.biases[layer-1]
        Z_activations = z_acts[layer]

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
            return self.backPropagate(None, desiredL1, layer-1, iterations, acts=acts, z_acts=z_acts)

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
        def function(index):
            self.backPropagate(data[i], output[i], self.calcs, iterations)
        threads = []
        for i in range(iterations):
            threads.append(threading.Thread(target=function, args=(i,)))
            threads[-1].start()
        for x in threads:
            x.join()
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
