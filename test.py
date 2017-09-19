import subprocess
import re

nCorrect = 0

for i in range(10):
	datasetDir = "Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/axpy","-i",datasetDir+"x.raw," + datasetDir+"y.raw,"+datasetDir+"a.raw","-e",datasetDir+"output.raw","-t","vector"])
	correct = re.search('"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 10 correct"
