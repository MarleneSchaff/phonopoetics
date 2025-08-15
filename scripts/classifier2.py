import argparse
import json
import gzip
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
#for results
from sklearn import metrics
import matplotlib.pyplot as plt
from torchmetrics.text import Perplexity
import pandas as pd

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("vecdata_in", help = "gz.jsonl with phonemes, stresses and authors")

	args = parser.parse_args()

	data = []
	with gzip.open(args.vecdata_in, "rt") as vd:
		for line in vd:
			jline = json.loads(line)
			data.append({"phones": jline["phones"], "stresses": jline["stresses"], "author_labels": jline["labels"]})

	phones = [d["phones"] for d in data]
#	stresses = [d["stresses"] for d in data]
	a_labels = [d["author_labels"] for d in data]

	labels = []
	for l in a_labels[0]:
		labels.append(l)

	phone_tensors = []
	for l in phones[0]:
		p_t = torch.tensor(l)
		phone_tensors.append(p_t)
#	phone_tensors = [torch.tensor(sample) for sample in phone_list]
#	stress_tensors = [torch.tensor(sample) for sample in stress]

	print("phone tensors created")

	padded_phones = pad_sequence(phone_tensors, batch_first=True)
#	padded_stresses = pad_sequence(stress_tensors)

	print("phone tensors padded")

	#concatenate in torch, either .cat() or .stack()
#	ps_vec = 
#	features = ps_vec

	df = pd.DataFrame(labels)
	df.columns = ["authors"]
	df.drop_duplicates(inplace=True)
	label_set = df["authors"].tolist()
	label_tensors = []
	for l in labels:
		for a in label_set:
			if l == a:
				auth_num = [label_set.index(a), 0]
				auth_tensor = torch.tensor(auth_num)
#				print(auth_tensor)
				label_tensors.append(auth_tensor) #.unsqueeze(0)


#	labels_tensor = torch.tensor(author_numbers, dtype=torch.long)
	#this needs to be one tensor with one value for each line
	print("labels encoded and tensor-ed")

#	print(label_tensors[1].numel())
#	print(padded_phones[1].numel())

	print(type(padded_phones))

#	phones_tensor = torch.stack(padded_phones)
	label_tensor = torch.stack(label_tensors)

	tensor_set = TensorDataset(padded_phones, label_tensor)
	train_dataset, test_dataset = torch.utils.data.random_split(tensor_set, [0.8, 0.2])
	print("train test split complete")
	train_batch = DataLoader(train_dataset, shuffle=True, batch_size=32)
	test_batch = DataLoader(test_dataset, batch_size = 32)

#	X_train, X_test, Y_train, Y_test = train_test_split(padded_phones, label_tensors, train_size=0.8, shuffle=True)

	length = len(padded_phones[0])
	print("tensor length")
	print(length)

	print("creating and training model now")

	class MLPClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
	super(MLPClassifier, self).__init__()
	self.fc1 = nn.Linear(input_size, hidden_size)
	self.relu = nn.ReLU()
	self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
	out = self.fc1(x)
	out = self.relu(out)
	out = self.fc2(out)
	return out

	model = MLPClassifier(input_size=56, hidden_size=32, num_classes=10)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)


#	class LinearClassifier(nn.Module):
#	def __init__(self, in_features, num_classes):
#	super(LinearClassifier, self).__init__()
#	self.linear = nn.Linear(in_features, num_classes)

#	def forward(self, x):
#	return self.linear(x)

#	in_features = 56 # Number of input features
#	num_classes = 10
#	model = LinearClassifier(in_features, num_classes)

#	output = model(input)

	print("model created")

	model.train()
	for n in range(5):
		for x_batch, y_batch in train_batch:
			y_pred = model(x_batch)
			loss = loss_fn(y_pred, y_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			torch.save(model, "classifier.pickle")
		print("epoch finished")
	#can retain dev set here to evaluate against
	#early stopping, if after 5 epochs no improvement then use last saved model

	model.eval()
	with torch.no_grad():
		for x_test, y_test in test_batch:
			y_pred_test = model(x_test)
			loss = loss_fn(y_pred_test, y_test)
			print(loss)

#confusion matrix
	actual = y_test
	predicted = y_pred
	confusion_matrix = metrics.confusion_matrix(actual, predicted)
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
	cm_display.plot()
	plt.show()

#how often is model correct
	accuracy = metrics.accuracy_score(actual, predicted)
#what percentage of positives are true positives
	precision = metrics.precision_score(actual, predicted)
#of positives, how many were predicted positive (also called sensitivity)
	recall = metrics.recall_score(actual, predicted)
#inverse, how many negs were correctly predicted
	specificity = metrics.recall_score(actual, predicted, pos_label = 0)
#F score
	f1_score = metrics.f1_score(actual, predicted)
#can print above assessments as dict
	print({"accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "F1 score": f1_score})

#perplexity score
#perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)
#with torch.no_grad():
#	output = model(#input, labels=input)
#logits = output.logits
#score = perplexity(preds=logits[:, :-1], target=input[:, :-1])
#print({"perplexity, ignoring padding": score.item()})
