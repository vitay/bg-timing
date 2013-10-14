all: 
	python GenerateTimecourse.py
	python GenerateUSMagnitude.py
	python GenerateNAcc.py
	python GenerateDips.py
	python GeneratePlots.py
