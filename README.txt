TEXT BASED INFORMATION RETRIEVAL ASSIGNMENT 1
Erik Gustav Rosvall & Sven Hermans

* dependencies:
- python3
- keras
- tensorflow (tensorflow-gpu is much faster)

* RUNNING the software: 
- in the terminal go this directory, i.e. the directory in which the file "lstm.py" is installed.
- $ python3 lstm.py [options]
- options: --ae [string]: filename of existing autoencoder model, will load this model instead of building new one
           --qa [string]: filename of existing question answerer model, will load this model instead of buidling new one
           --e [int]:  number of epochs, default 1
           --ld1 [int]: latent dimension 1, default 140
           --ld2 [int]: latent dimension 2, default 50
           --b [int]: batch size, default 32
- some examples: 
1. python3 lstm.py
2. python3 lstm.py --ae autoencoder.h5 --qa question_answerer.h5
3. python3 lstm.py --e 20 --ld1 300 --ld2 100
4. python3 lstm.py --b 

* autoencoder.h5 contains the best autoencoder model so far
* question_answerer.h5 contains the best question answerer model so far
* Use example 2 above to evaluate our best models