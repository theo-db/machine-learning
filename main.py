import network as net
import numpy, random

print("reading files")

Ifile = open("train-images.idx3-ubyte", "rb")
Lfile = open("train-labels.idx1-ubyte", "rb")

Ifile.read(4)
Lfile.read(8)
items = int.from_bytes(Ifile.read(4), "big")
rows = int.from_bytes(Ifile.read(4), "big")
columns = int.from_bytes(Ifile.read(4), "big")
size = rows * columns

network = net.Network([size, 30, 10])
#network = net.loadNetwork("number.txt")

images = []
numbers = []

for i in range(items):
    num = int.from_bytes(Lfile.read(1), "big")
    numbers.append(num)
    image = [int.from_bytes(Ifile.read(1),"big") for j in range(size)]
    images.append(numpy.array(image, numpy.float64) / 256)
Ifile.close()
Lfile.close()

expected = []
for i in range(items):
    arr = numpy.zeros(10, numpy.float64)
    arr[numbers[i]] = 1.0
    expected.append(arr)

#now onto the training part
print("training network")

iterations = 10
count = 0
for i in range(6000):
    indices = random.sample(range(items), iterations)
    data = [images[j] for j in indices]
    desired = [expected[j] for j in indices]
    network.train(data, desired, iterations)
    count += 1
    if count % 100 == 0:
        print("{}%".format(100*i*iterations/items))

print("saving network")
net.saveNetwork(network, "number.txt")


#for i in range(100):
#    index = i
#    network.use(images[index])
#    r = network.getResult()
#    num = numbers[index]
#    print("{} : {}, {}".format(num,r,r==num))

print("testing network")

print("reading files")

TIfile = open("t10k-images.idx3-ubyte", "rb")
TLfile = open("t10k-labels.idx1-ubyte", "rb")

TIfile.read(4)
TLfile.read(8)
items = int.from_bytes(TIfile.read(4), "big")
TIfile.read(8)

images = []
numbers = []

for i in range(items):
    num = int.from_bytes(TLfile.read(1), "big")
    numbers.append(num)
    image = [int.from_bytes(TIfile.read(1),"big") for j in range(size)]
    images.append(numpy.array(image, numpy.float64) / 256)
TIfile.close()
TLfile.close()

expected = []
for i in range(items):
    arr = numpy.zeros(10, numpy.float64)
    arr[numbers[i]] = 1.0
    expected.append(arr)

count = 0
successes = 0
for i in range(items):
    network.use(images[i])
    r = network.getResult()
    num = numbers[i]
    count += 1
    if r==num:
        successes += 1
    if count % 100 == 0:
        print(100*successes/count)

print("final score: {}%".format(round(100*successes/count, 1)))
        
    
    

    

