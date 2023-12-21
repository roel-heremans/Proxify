Setting up the virtual environment
## deactivate any environment that may be active
$conda deactivate

## creating virtual environment called 'dbconnect'
$conda create --name dbconnect python=3.8

## activate the 'dbconnect' environment
$conda activate dbconnect

## install the necessary pacakages mentioned in the requirements txt-file
pip install -r requirements.txt
