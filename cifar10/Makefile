HOME := /home/alu
ANACONDA := $(HOME)/anaconda3
PY := python
WORK := $(HOME)/work
#####!!!!!!!!!!! NEVER USE PATH!!!!!! PATH := $(WORK)/tf/honk

NN_MODEL := demo_model
#NN_MODEL := resnet50
#NN_MODEL := alexnet
#NN_MODEL := vgg16_bn
##NN_MODEL := SeResNet3
#NN_MODEL := res8
#NN_MODEL := cnn-trad-pool2

## Global configuration parameters
EPOCH := 20
BATCH := 256
#BATCH := 64
CHKINIT := 'output/checkpoint/00003000_model.pth'

LR := 0.01

#IMGNET := /media/alu/bigdata/image/ILSVRC2015/Data/CLS-LOC
## SSD
#IMGNET := /media/alu/MX500SSD/image/ILSVRC2015/Data/CLS-LOC
IMGNET := /media/alu/L5LITESSD/image/ILSVRC2015/Data/CLS-LOC
LOGFILE := ./output/log.train.txt
STARTDL := START
ENDDL := ZZZZZ

.PHONY: help train infer \
	clean-pyc all all2 \
	autobuild-dev autobuild-stable

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  train      to train the neural network"
	@echo "  serve      to serve the documentation on the localhost (8000)"


train:
	${PY} train.py --name "demo" --model ${NN_MODEL}


clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
