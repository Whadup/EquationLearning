import torch.optim as optim
import os
import shutil
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn
import numpy as np
import argparse
from formula_data import SingleData
from equation_encoder import NetSmall, NetLarge

def convert_to_la(img):
	return img.convert(mode='LA')


def take_alpha_channel(img):
	return img[1].unsqueeze(0)


NUM_TOKENS = 1500

class PretrainEquationData(torch.utils.data.Dataset):
	""" This class is a torch dataset, which gives a pair of real arxiv formulas.
		If the label is 1 the two formulas occur in the same paper.
	"""

	# isProcessed determines if the Dataset was already built. Then it will be loaded from a file.
	def __init__(self, name, root="", is_processed=True, pairs_per_sample=1, one_per_class=False):
		self.SIM_LABEL = 1
		self.DISSIM_LABEL = 0
		self.labels = [self.DISSIM_LABEL, self.SIM_LABEL]
		self.pairs_per_sample = pairs_per_sample
		self.one_per_class = one_per_class
		self.num_labels = 1500
		transform = transforms.Compose(
				[convert_to_la,
				 transforms.CenterCrop((32, 333)),
				 transforms.ToTensor(),
				 take_alpha_channel
				 ])
		# Get data from directory structure
		self.data = []
		self.dataset = torchvision.datasets.ImageFolder(root=root, transform=transform, loader=img_loader)
		self.tokens = [None for i in range(len(self.dataset))]
		if is_processed:
			import pickle
			self.data = pickle.load(open("imageFolder.pickle","rb"))
		for index in range(len(self.dataset)):
			path, target = self.dataset.samples[index]
			if not is_processed:
				sample = self.dataset.loader(path)
				if self.dataset.transform is not None:
					sample = self.dataset.transform(sample)
				self.data.append(sample)
			if self.tokens[index] is None:
				dirname = os.path.dirname(path)
				pname = os.path.basename(path)[:-3]+"csv"
				with open(os.path.join(dirname,pname),"r") as f:
					l = f.readline().strip()
					self.tokens[index] = [int(x) for x in l.strip().split(",") if len(x) and int(x)<self.num_labels]
		if not is_processed:
			import pickle
			pickle.dump(self.data,open("imageFolder.pickle","wb"))
			# self.index_label_tensor = self.create_pairs(digit_indices)
			# torch.save(self.index_label_tensor, self.store_file)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.data[index]
		if len(self.tokens[index]):
			if np.random.uniform()<0.5:
				return sample, torch.LongTensor([np.random.choice(self.tokens[index])]), self.SIM_LABEL
		return sample, torch.LongTensor([np.random.choice(self.num_labels)]), self.DISSIM_LABEL 

class PretrainAbstractData(torch.utils.data.Dataset):
	""" This class is a torch dataset, which gives a pair of real arxiv formulas.
		If the label is 1 the two formulas occur in the same paper.
	"""

	# isProcessed determines if the Dataset was already built. Then it will be loaded from a file.
	def __init__(self, name, root="", is_processed=True, pairs_per_sample=1, one_per_class=False):
		self.SIM_LABEL = 1
		self.DISSIM_LABEL = 0
		self.pairs_per_sample = pairs_per_sample
		self.one_per_class = one_per_class
		self.num_labels = 186
		transform = transforms.Compose(
				[convert_to_la,
				 transforms.CenterCrop((32, 333)),
				 transforms.ToTensor(),
				 take_alpha_channel
				 ])
			# Get data from directory structure
		self.dataset = torchvision.datasets.ImageFolder(root=root, transform=transform, loader=img_loader)
		self.data = []
		self.tokens = [None for i in range(len(self.dataset))]
		if is_processed:
			import pickle
			self.data = pickle.load(open("imageFolder.pickle","rb"))
		self.neg_probs = [0.0 for i in range(self.num_labels)]
		
		for index in range(len(self.dataset)):
			path, target = self.dataset.samples[index]
			if not is_processed:
				sample = self.dataset.loader(path)
				if self.dataset.transform is not None:
					sample = self.dataset.transform(sample)
				self.data.append(sample)
			if self.tokens[index] is None:
				dirname = os.path.dirname(path)
				# pname = os.path.basename(path)[:-3]+"csv"
				p = os.path.join(dirname, "keywords.txt")
				with open(p, "r") as f:
					f.readline()
					f.readline()
					l = f.readline().strip()
					tokens = [int(x) for x in l.strip().split(";") if len(x) and int(x)<self.num_labels]
					for t in tokens:
						self.neg_probs[t]+=1
					self.tokens[index] = tokens
		self.neg_probs = np.array([t ** 0.75 for t in self.neg_probs])
		self.neg_probs/=np.sum(self.neg_probs)
		print(self.neg_probs)
		print(sorted(self.neg_probs))
		if not is_processed:
			import pickle
			pickle.dump(self.data, open("imageFolder.pickle","wb"))
			# self.index_label_tensor = self.create_pairs(digit_indices)
			# torch.save(self.index_label_tensor, self.store_file)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.data[index]

		if len(self.tokens[index]):
			if np.random.uniform()<0.5:
				return sample, torch.LongTensor([np.random.choice(self.tokens[index])]), self.SIM_LABEL
		return sample, torch.LongTensor([np.random.choice(self.num_labels, p=self.neg_probs)]), self.DISSIM_LABEL

from PIL import Image
def img_loader(path):
	return Image.open(path)

def train(batch_size, learning_rate, momentum, weight_decay, epochs,
		  scheduler_patience, with_dot_product, dataset, architecture, ex, pretrained_weights):
	if architecture == 'small':
		net = NetSmall(with_dot_product=with_dot_product)
	elif architecture == 'large':
		net = NetLarge(with_dot_product=with_dot_product)
	else:
		raise AssertionError()
	if pretrained_weights is not None:
		net.load_state_dict_from_path(pretrained_weights)
		if architecture == 'large':
			net.use_batch_norm = False
		ex.add_resource(pretrained_weights)
	print(dataset)
	if dataset=="abstract":
		dataset = PretrainAbstractData(name="train",root="weak_data_train", is_processed=True)
	elif dataset=="equation":
		dataset = PretrainEquationData(name="train",root="weak_data_train", is_processed=True)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
		sampler=torch.utils.data.RandomSampler(dataset))

	num_batches = len(trainloader)
	print(num_batches)
	print(len(dataset))

	loss_output_interval = 100

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = net.to(device)
	embeddings = torch.nn.Embedding(dataset.num_labels, 64).to(device)
	criterion = torch.nn.BCEWithLogitsLoss()
	# crterion = torch.nn.MSELoss()
	# optimizer = optim.SGD(list(net.parameters())+list(embeddings.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
	optimizer = optim.Adam(list(net.parameters())+list(embeddings.parameters()), lr=learning_rate)
	# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
	print(criterion)
	for epoch in range(epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader):
			# get the inputs
			in1, in2, labels = data
			if in1.size()[0] != batch_size:
				# print("skip")
				continue
			
			in1 = in1.to(device)
			in2 = in2.to(device)
			labels = labels.to(device)
			

			# forward + backward + optimize
			h = net.forward(in1)
			y = embeddings(in2)
			# print(h.shape,y.shape)
			dist = torch.bmm(h.view(-1, 1, 64), y.view(-1, 64, 1)).view(-1)
			dist.to(device)
			# dist = torch.nn.functional.tanh(dist)
			labels = labels.float()
			# # print(dist * labels)
			loss = criterion(dist, labels)
			#print(torch.histc(torch.nn.functional.sigmoid(dist).cpu(),bins=7,min=0,max=0),loss.item())
			#     # print(loss)
			# else:
			
			# loss = criterion(dist, labels)
			# print(torch.bincount(torch.max(0,-torch.min(dist).floor().int())+dist.round().int()).cpu().numpy(),loss.item())
			# print(torch.histc(dist.cpu(),bins=7),loss.item())
			# print(loss)
			loss.backward()
			optimizer.step()
			# zero the parameter gradients
			optimizer.zero_grad()

			# print statistics
			running_loss += loss.item()
			if i % loss_output_interval == loss_output_interval - 1:
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / loss_output_interval))
				ex.log_scalar("training.loss", running_loss / loss_output_interval)
				running_loss = 0.0
				#if i+1==300:
				#    break
		print('[%d, %5d] loss: %.3f' %
			  (epoch + 1, i + 1, running_loss / (i % loss_output_interval)))
		ex.log_scalar("training.loss", running_loss / (i % loss_output_interval))
		net.save_checkpoint(epoch)
		ex.add_artifact(net.checkpoint_string.format(epoch))
		if architecture == "large" and epoch == 0:
			net.kill_batch_norm()
		scheduler.step()
		num_batches_in_last_interval = len(trainloader) % loss_output_interval
		last_loss = running_loss / num_batches_in_last_interval
		ex.info["loss after epoch {}".format(epoch + 1)] = last_loss
	print('Finished Training')
	net.to("cpu")
	net.save()
	print("Saved")

if __name__ == "__main__":
	print("Did you mean to execute pretrain_experiment.py?")