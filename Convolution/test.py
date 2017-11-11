import subprocess
import re

nCorrect = 0

for i in range(7):
	datasetDir = "./Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/Convolution","-i",datasetDir+"input0.ppm,"+ datasetDir+"input1.raw","-e",datasetDir+"output.ppm","-t","image"])
	correct = re.search('"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 7 correct"
