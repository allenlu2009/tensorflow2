HOME := /home/alu
ANACONDA := $(HOME)/anaconda3
ACT := $(ANACONDA)/bin/activate
VENV := tensorflow
PY := python
PY2 := /usr/bin/env python
SHELL := /bin/bash
WORK := $(HOME)/work
#####!!!!!!!!!!! NEVER USE PATH!!!!!! PATH := $(WORK)/tf/honk

#WORDS := yes no up down left right on off stop go

.PHONY: help train infer \
	clean-pyc all all2 \
	autobuild-dev autobuild-stable

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  train      to train the neural network"
	@echo "  infer      to infer using the trained neural network"
	@echo "  check      to run a check for frequent markup errors"
	@echo "  serve      to serve the documentation on the localhost (8000)"

train:
	${PY} mnist_tf2.py

infer:
	${PY} train.py --idataset ${INF_DATASET} --net ${NN_MODEL} --chkinit ${CHKINIT} --mode eval

plot:
	${PY} extract2marks.py -i ${LOGFILE} -o tst.log -s ${STARTDL} -z ${ENDDL}
	${PY} plot_parse.py -d ${TRN_DATASET} -n ${NN_MODEL}

chop:
	${PY} extract2marks.py -i ${LOGFILE} -o tst.log -s ${STARTDL} -z ${ENDDL}

infer2:
	$(SHELL) -c "source ${ACT} ${VENV} ; ${PY} -m utils.train --wanted_words ${WORDS} --dev_every 1 --n_labels 12  --schedule 3000 6000 --net ${MODEL} --input_file ./model/model.pt --mode eval" 

all:
	$(SHELL) -c "source ${ACT} ${VENV} ; ${PY} -m utils.train --wanted_words ${WORDS} --dev_every 1 --n_labels 12 --n_epochs 26 --weight_decay 0.00001 --lr 0.1 0.01 0.001 --schedule 3000 6000 --net ${MODEL}"

all2:
	$(SHELL) -c "source ${ACT} ${VENV} ; ${PY} -m utils.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 26 --weight_decay 0.00001 --lr 0.1 0.01 0.001 --schedule 3000 6000 --net res26-narrow"


clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
