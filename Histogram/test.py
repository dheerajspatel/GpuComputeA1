import subprocess
import re

nCorrect = 0

for i in range(5):
	datasetDir = "./Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/Histogram","-i",datasetDir+"input.raw","-e",datasetDir+"output.raw","-t","integral_vector"])
	correct = re.search('"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 5 correct"
