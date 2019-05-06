"""
All the models and training methods used for unsupervised learning of
embeddings for equations. Based on torch.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from formula_data import PairData, TripleData

def gelu(x):
	"""
	Implementation of the gelu activation function.
	For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
	0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	Also see https://arxiv.org/abs/1606.08415
	"""
	import math
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class NetSmall(nn.Module):
	"""
	A small CNN, the base class for all models
	"""

	def __init__(self, with_dot_product=False):
		# constant initializing
		self.checkpoint_string = "equation_encoder_small_weights_checkpoint{}.pt"
		self.with_dot_product = with_dot_product
		self.save_path = "si_bear_weights.pt"
		
		super(NetSmall, self).__init__()

		self.euclid = torch.nn.PairwiseDistance(p=2)
		# pooling layers
		self.pool2x4 = nn.MaxPool2d((2, 4))
		self.pool3x3 = nn.MaxPool2d((3, 3))
		self.very_much_pool = nn.AvgPool2d((1, 15))

		in_ch1 = 1
		out_ch1 = 32
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_ch1, out_ch1, 3),
			# nn.BatchNorm2d(out_ch1),
		)

		in_ch2 = out_ch1
		out_ch2 = 32
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_ch2, out_ch2, 5),
			# nn.BatchNorm2d(out_ch2, affine=False),
		)

		in_ch_single = out_ch2
		out_ch_single = 32
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_ch_single, out_ch_single, 3),
			# nn.BatchNorm2d(out_ch_single),
		)
		self.flatten = 5 * 32
		self.fc3 = nn.Linear(5 * 32, 64)
		self.lin_norm = nn.BatchNorm1d(64, affine=False, momentum=0.01)
		self.fc4 = nn.Linear(64, 64)

	def forward_plain(self,x):
		x = self.conv1(x)
		x = gelu(x)
		# print(x.shape)
		x = self.pool2x4(x)
		x = self.conv2(x)
		# print(x.shape)
		x = gelu(x)
		x = self.pool2x4(x)
		x = self.conv3(x)
		# print(x.shape)

		x = gelu(x)
		x = self.pool3x3(x)
		# print(x.shape)

		x = x.view(-1, self.flatten)
		x = self.fc3(x)
		return x
	
	def forward(self, x):

		# x = gelu(self.fc3(x))
		# print(torch.norm(x,dim=1))
		# x = self.lin_norm(x)
		# x = torch.tanh(self.fc4(x))/8.0
		x = self.forward_plain(x)
		norm = torch.norm(x,dim=1,keepdim=True) + 0.00001
		x = x.div(norm.expand_as(x))
		return x

	def forward2(self, in1, in2):
		"""process two inputs and return the distance/similarity"""
		out1 = self.forward(in1)
		out2 = self.forward(in2)

		if self.with_dot_product:
			result = torch.bmm(out1.view(-1, 1, 64), out2.view(-1, 64, 1)).view(-1)
			return result
		return self.euclid(out1, out2)

	def forward3(self, in1, in2, in3):
		"""Process three inputs, return embeddings or similarities"""
		out1 = self.forward(in1)
		out2 = self.forward(in2)
		out3 = self.forward(in3)
		if self.with_dot_product:
			dist_sim = torch.bmm(out1.view(-1, 1, 64), out2.view(-1, 64, 1)).view(-1)
			dist_dissim = torch.bmm(out1.view(-1, 1, 64), out3.view(-1, 64, 1)).view(-1)
			dist_dissim2 = torch.bmm(out2.view(-1, 1, 64), out3.view(-1, 64, 1)).view(-1)
			return dist_sim, torch.max(dist_dissim, dist_dissim2)
		return out1, out2, out3

	def save(self):
		"""store the final model"""
		torch.save(self.state_dict(), self.save_path)

	def load(self):
		"""load a trained model"""
		self.load_state_dict(torch.load(self.save_path, map_location='cpu'))

	def save_checkpoint(self, epoch):
		"""save a checkpoint"""
		torch.save(self.state_dict(), self.checkpoint_string.format(epoch))

	def load_checkpoint(self, epoch):
		"""restore a checkpoint"""
		self.load_state_dict(torch.load(self.checkpoint_string.format(epoch), map_location='cpu'))

	def load_state_dict_from_path(self, path):
		"""load a model from a wrong path"""
		self.load_state_dict(torch.load(path, map_location='cpu'))


class NetLarge(NetSmall):
	"""
	A larger CNN, the base class for all models
	"""

	def __init__(self, with_dot_product=False):
		# constant initializing


		super(NetLarge, self).__init__(with_dot_product=with_dot_product)
		self.checkpoint_string = "equation_encoder_large_weights_checkpoint{}.pt"
		self.with_dot_product = with_dot_product
		self.save_path = "si_bear_large_weights.pt"
		self.euclid = torch.nn.PairwiseDistance(p=2)
		# pooling layers
		self.pool2x4 = nn.MaxPool2d((2, 4))
		self.pool3x3 = nn.MaxPool2d((1, 3))
		self.very_much_pool = nn.AvgPool2d((1, 15))

		in_ch1 = 1
		out_ch1 = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_ch1, out_ch1, 3),
			nn.ReLU(),
			nn.Conv2d(out_ch1, out_ch1, 3),
			# nn.BatchNorm2d(out_ch1),
		)

		in_ch2 = out_ch1
		out_ch2 = 64
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_ch2, out_ch2, (3,5)),
			nn.ReLU(),
			nn.Conv2d(out_ch2, out_ch2, (3,5)),
			# nn.BatchNorm2d(out_ch2, affine=False),
		)

		in_ch_single = out_ch2
		out_ch_single = 64
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_ch_single, out_ch_single, 3),
			nn.ReLU(),
			nn.Conv2d(out_ch_single, out_ch_single, 3),
			# nn.BatchNorm2d(out_ch_single),
		)
		self.flatten = 64 * 4
		self.fc3 = nn.Linear(self.flatten, 64)
		self.lin_norm = nn.BatchNorm1d(64, affine=False, momentum=0.1)
		self.fc4 = nn.Linear(64, 64)
		self.use_batch_norm = True

	def forward(self, x):
		x = super(NetLarge,self).forward_plain(x)
		if self.use_batch_norm:
			x = self.lin_norm(x)
			# print("bn")

		x = self.fc4(x)
		norm = torch.norm(x,dim=1,keepdim=True) + 0.00001
		x = x.div(norm.expand_as(x))
		return x

	def kill_batch_norm(self):
		if self.use_batch_norm:
			print(self.lin_norm.running_mean)
			print(self.lin_norm.running_var)
			print(1.0/torch.sqrt(self.lin_norm.running_var + self.lin_norm.eps))
			self.fc4.weight.data = torch.matmul(self.fc4.weight.data, torch.diag(1.0/torch.sqrt(self.lin_norm.running_var + self.lin_norm.eps)))
			self.fc4.bias.data -= torch.matmul(self.fc4.weight.data, self.lin_norm.running_mean)
			print("DIE YOU FUCKING BATCH NORM LAYER AND ROT IN HELL!!!")
		self.use_batch_norm = False



HISTOGRAM_STRINGS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

def print_histogram(dist, remainder="\b", compute_max=None, compute_min=None):
	"""print a histogram of values"""
	try:
		dist = dist.cpu()
		hist_max = compute_max if compute_max is not None else torch.max(dist).item()
		hist_min = compute_min if compute_min is not None else torch.min(dist).item()
		hist = torch.histc(dist.cpu(), bins=15, min=hist_min, max=hist_max).detach().numpy()
		m = np.max(hist)
		hist = np.round(8*hist/m).astype(int)
		print("{:.2f} ".format(hist_min) +
			"".join([2*HISTOGRAM_STRINGS[x] for x in hist]) +
			" {:.2f}".format(hist_max) +
			" " + str(remainder))
	# pragma pylint: disable=broad-except
	except Exception as exception:
		print(exception)


def train(batch_size, learning_rate, epochs,
			with_dot_product, dataset, eval_dataset, architecture, ex, pretrained_weights, triples):
	"""
	train a model
	"""

	if architecture == 'small':
		net = NetSmall(with_dot_product=with_dot_product)
		net.train()
	elif architecture == 'large':
		net = NetLarge(with_dot_product=with_dot_product)
	else:
		raise AssertionError()

	if pretrained_weights is not None:
		print("Loading weights")
		net.load_state_dict_from_path(pretrained_weights)
		# net.fc3.weight.data *= 0.2
		ex.add_resource(pretrained_weights)

	if triples:
		dataset = TripleData(name=dataset)
	else:
		dataset = PairData(name=dataset, with_dot_product=with_dot_product)

	eval_dataset = TripleData(name=eval_dataset)

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
						shuffle=False, sampler=torch.utils.data.RandomSampler(dataset))

	loss_output_interval = 100

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = net.to(device)

	zeros = torch.zeros(batch_size).to(device)
	ones = torch.ones(batch_size).to(device)

	if triples:
		if with_dot_product:
			criterion = torch.nn.MarginRankingLoss(margin=1, reduction="none")
		else:
			criterion = torch.nn.TripletMarginLoss(swap=True)
	else:
		if with_dot_product:
			# criterion = torch.nn.BCEWithLogitsLoss()
			criterion = torch.nn.MarginRankingLoss(margin=1, reduction="none")
		else:
			criterion = torch.nn.HingeEmbeddingLoss(margin=1)

	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

	euclid = torch.nn.PairwiseDistance()

	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader):
			if triples:
				in1, in2, in3 = data
				in1 = in1.to(device)
				in2 = in2.to(device)
				in3 = in3.to(device)
				if with_dot_product:
					dist_sim, dist_dissim = net.forward3(in1, in2, in3)
					loss = criterion(dist_sim, dist_dissim, ones[:in1.shape[0]])
					loss = loss * loss # squared hinge instead of hinge
					loss = torch.mean(loss)
					if i % loss_output_interval == loss_output_interval - 1:
						# print(dist_sim)
						# print(dist_dissim)
						print_histogram(dist_sim.detach(), compute_max=1, compute_min=-1)
						print_histogram(dist_dissim.detach(), loss.item(), compute_max=1, compute_min=-1)
				else:
					out1, out2, out3 = net.forward3(in1, in2, in3)

					dist_sim = euclid(out1, out2)
					dist_dissim = torch.min(euclid(out1, out3), euclid(out2, out3))
					loss = torch.max(zeros[:in1.shape[0]], 1.0-dist_dissim/(dist_sim+1.0))
					loss = torch.mean(loss)
					# loss = criterion(out1, out2, out3)
					if i % loss_output_interval == loss_output_interval - 1:
						# dist_sim = euclid(out1.detach(), out2.detach())
						# dist_dissim = euclid(out1.detach(), out3.detach())
						print_histogram(dist_sim, compute_max=2, compute_min=0)
						print_histogram(dist_dissim, loss.item(), compute_max=2, compute_min=0)
			else:
				in1, in2, labels = data
				in1 = in1.to(device)
				in2 = in2.to(device)
				labels = labels.to(device).float()
				dist = net.forward2(in1, in2)
				if with_dot_product:
					#loss = criterion(dist, labels)
					loss = criterion(dist, zeros[:in1.shape[0]], labels)
					loss = loss * loss # squared hinge instead of hinge
					loss = torch.mean(loss)
					if i % loss_output_interval == loss_output_interval - 1:
						print_histogram(labels * dist, compute_max=1, compute_min=-1)
				else:
					loss = criterion(dist, labels)
					if i % loss_output_interval == loss_output_interval - 1:
						print_histogram((labels-0.5) * dist, compute_max=1, compute_min=-1)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			# print statistics
			running_loss += loss.item()
			if i % loss_output_interval == loss_output_interval - 1:
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / loss_output_interval))
				ex.log_scalar("training.loss", running_loss / loss_output_interval)
				running_loss = 0.0
		print('[%d, %5d] loss: %.3f' %
			  (epoch + 1, i + 1, running_loss / (i % loss_output_interval)))
		ex.log_scalar("training.loss", running_loss / (i % loss_output_interval))
		net.save_checkpoint(epoch)
		ex.add_artifact(net.checkpoint_string.format(epoch))
		scheduler.step()
		if architecture == "large" and epoch == 0:
			net.kill_batch_norm()
		# evaluation.eval_loss_triple(net, eval_dataset, with_dot_product)
		num_batches_in_last_interval = len(trainloader) % loss_output_interval
		last_loss = running_loss / num_batches_in_last_interval
		ex.info["loss after epoch {}".format(epoch + 1)] = last_loss
	print('Finished Training')
	net.to("cpu")
	net.save()
	print("Saved")


if __name__ == "__main__":
	raise NotImplementedError()
