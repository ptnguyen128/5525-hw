
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

def plotLoss(k, title):
	log_loss = []
	runs = range(5)

	for run in runs:
		pickle_off = open("./pickle/SoftplusLoss/loss_run_%s_k_%s.pickle" % (run,k), 'rb')
		log_loss.append(pickle.load(pickle_off))
		pickle_off.close()

	plt.plot(log_loss[0], color='blue', marker='o', label='Run 1', markevery=10)
	plt.plot(log_loss[1], color='green', marker='x', label='Run 2', markevery=15)
	plt.plot(log_loss[2], color='red', marker='o', label='Run 3', markevery=20)
	plt.plot(log_loss[3], color='orange', marker='x', label='Run 4', markevery=25)
	plt.plot(log_loss[4], color='purple', marker='o', label='Run 5', markevery=30)
	plt.title(title)
	plt.legend()
	plt.ylabel("Log Loss")
	plt.xlabel("Iterations")
	plt.show()

if __name__ == "__main__":
	ks = [1, 20,200,1000,2000]
	for k in ks:
		plotLoss(k, "Log Loss vs. #Iterations (k=%s)" % k)
