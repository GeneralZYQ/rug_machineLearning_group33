
#Produce the input features

import math 

fotraning = open('ae.train', 'r+')
allmanFeatures = []
oneManFeatures = []
while True:

    line = fotraning.readline()

    if not line:
        break;

    components = line.split(' ')
    del components[-1]

    if len(components) == 13:
        del components[-1]
    
    if float(components[0]) != 1.0:
        oneManFeatures.append(components)
    else:
        allmanFeatures.append(oneManFeatures);
        oneManFeatures = [];


# print (len(allmanFeatures[1]))

trainingLabels = []
for i in range(0,270):
    featureIndex = math.ceil(i / 270.0)
    trainingLabels.append(featureIndex)



fotesting = open('ae.train', 'r+')
alltestFeatures = []
oneTestFeatures = []
while True:

    line = fotesting.readline()

    if not line:
        break;

    components = line.split(' ')
    del components[-1]

    if len(components) == 13:
        del components[-1]
    
    if float(components[0]) != 1.0:
        oneTestFeatures.append(components)
    else:
        alltestFeatures.append(oneTestFeatures);
        oneTestFeatures = [];

testingLabels = []
blockLengthes = [31, 35, 88, 44, 29, 24, 40, 50, 29];

total = 0
for j in range(0,len(blockLengthes)):
    num = blockLengthes[j];
    for k in range(total, total + num):
        testingLabels.append(j + 1);

print (len(testingLabels))

    

    
    
