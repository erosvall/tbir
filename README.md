TEXT BASED INFORMATION RETRIEVAL ASSIGNMENT
Erik Gustav Rosvall & Sven Hermans

A Visual Question Answerer

* dependencies:
- python3
- keras
- tensorflow (tensorflow-gpu is much faster)

* RUNNING the software: 
- in the terminal go this directory, i.e. the directory in which the file "vqa.py" is installed.
- $ python3 vqa.py [-h] [--load LOAD] [--test TEST] [--q Q] [--e E] [--ld1 LD1]
              [--b B] [--drop DROP] [--aeweight AEWEIGHT] [--wups]
              [--textonly] [--visualonly] [--improve] [--checkpoint]
- optional arguments:
			  -h, --help           show this help message and exit
			  --load LOAD          Filename of existing model, default None
			  --test TEST          Filename of test data, default qa.894.raw.test.txt
			  --q Q                Pose one question for the visual question answerer
			  --e E                Number of epochs, default 1
			  --ld1 LD1            Latent dimension 1, default 512
			  --b B                Batch size, default 32
			  --drop DROP          Dropout percentage, default 0.5
			  --aeweight AEWEIGHT  Weight of the autoencoder loss function compared to the
			                       answer loss function, default 1.0
			  --wups               Compute the WUPS Score
			  --textonly           Ignore the images
			  --visualonly         Without autoencoder
			  --improve            Further train the loaded model
			  --checkpoint         Save at every epoch
- some examples: 
1. python3 vqa.py
2. python3 vqa.py --load model_full.h5 --q "What is on the table in the image555 ?"
3. python3 vqa.py --load model_full.h5 --test data.txt --wups
4. python3 vqa.py --load model_text.h5 --test data.txt --wups --textonly
5. python3 vqa.py --load model_visual.h5 --test data.txt --wups --visualonly
6. python3 vqa.py --e 20 --ld1 300 --b 512 --drop 0.7
7. python3 vqa.py --e 20 --ld1 300 --b 512 --drop 0.7 --textonly
8. python3 vqa.py --e 20 --ld1 300 --b 512 --drop 0.7 --visualonly

* model_full.h5 contains the full model (visual & text input -> question & answer output)
* model_text.h5 contains the text model (text input -> question & answer output)
* model_visual.h5 contains the visual model (visual & text input -> answer output)
* Use examples 3-5 above to evaluate these models