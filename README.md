TEXT BASED INFORMATION RETRIEVAL ASSIGNMENT
Erik Gustav Rosvall & Sven Hermans

A Visual Question Answerer

* dependencies:
- python3
- keras
- tensorflow (tensorflow-gpu is much faster)

* RUNNING the software: 
- in the terminal go this directory, i.e. the directory in which the file "vqa.py" is installed.
- $ python3 vqa.py [-h] [--load LOAD] [--test TEST] [--e E] [--ld1 LD1] [--b B]
              [--drop DROP] [--aeweight AEWEIGHT] [--wups] [--textonly]
              [--visualonly] [--improve] [--checkpoint]
- optional arguments:
			  -h, --help           show help message and exit
			  --load LOAD          Filename of existing model, default None
			  --test TEST          Filename of test data, default qa.894.raw.test.txt
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
2. python3 vqa.py --load full_model.h5 --test data.txt --wups
3. python3 vqa.py --load text_model.h5 --test data.txt --wups --textonly
4. python3 vqa.py --load visual_model.h5 --test data.txt --wups --visualonly
5. python3 vqa.py --e 20 --ld1 300 --b 512 --drop 0.7
6. python3 vqa.py --e 20 --ld1 300 --b 512 --drop 0.7 --textonly
7. python3 vqa.py --e 20 --ld1 300 --b 512 --drop 0.7 --visualonly

* full_model.h5 contains the full model (visual & text input -> question & answer output)
* text_model.h5 contains the text model (text input -> question & answer output)
* visual_model.h5 contains the visual model (visual & text input -> answer output)
* Use examples 2-4 above to evaluate these models