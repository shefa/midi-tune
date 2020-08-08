# installs python requirements and initializes weights and biases
pip3 install --update pip
pip3 install -r requirements.txt
pip3 install tensorflowjs --no-deps
~/.local/bin/wandb init