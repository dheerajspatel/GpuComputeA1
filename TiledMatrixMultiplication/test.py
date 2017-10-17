import subprocess
import re

nCorrect = 0

for i in range(10):
	datasetDir = "Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/TiledMatrixMultiplication","-i",datasetDir+"input0.raw," + datasetDir+"input1.raw","-e",datasetDir+"output.raw","-t","matrix"])
	correct = re.search('"correctq": true',result) != None
	aDimMatch = re.search('"The dimensions of A are (\\d+) x (\\d+)"',result)
	bDimMatch = re.search('"The dimensions of B are (\\d+) x (\\d+)"',result)

	totalFLOPS = 2 * int(aDimMatch.group(1)) * int(aDimMatch.group(2)) * int(bDimMatch.group(2))	
	tiledMatch = re.search('\\{.*"elapsed_time": (\\d+).*"message": "Performing basic tiled computation"',result)
	tiledTime = int(tiledMatch.group(1)) / 1e9

	print "basic tiling: " + str(totalFLOPS) + " total FLOPs / " + str(tiledTime) + " s = " + str(totalFLOPS / tiledTime / 1e9) + " GFLOPS / s"

	multitiledMatch = re.search('\\{.*"elapsed_time": (\\d+).*"message": "Performing multi-tiled computation"',result)
	multitiledTime = int(multitiledMatch.group(1)) / 1e9
	print "multi-tiling: " + str(totalFLOPS) + " total FLOPs / " + str(multitiledTime) + " s = " + str(totalFLOPS / multitiledTime / 1e9) + " GFLOPS / s"

	if correct:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 10 correct"
