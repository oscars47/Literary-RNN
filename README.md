# Literary-RNN
## Description
A repository for my recurrent neural network literature generation projects. My aim in these projects is to have fun exploring machine learning and literature, and I wish to make my discoveries available to anyone else who is curious! Check out the releases for updated versions of my RNNs!

You can interact with my RNN model Thinking Parrot at my HuggingFace space: https://huggingface.co/spaces/oscars47/Thinking_Parrot_1.1.0, https://huggingface.co/spaces/oscars47/Thinking_Parrot_1.0.1, and https://huggingface.co/spaces/oscars47/thinking_parrot_reading_club_redux.  My code is available at: https://github.com/oscars47/Literary-RNN 

See the Twitter posts by Thinking Parrot here: https://twitter.com/parrot_thinking/with_replies

## Instructions for use:
1. First, fork this repo.
2. Then clone  using ```git clone <url>```.
3. The folder model_v0.0.1 contains files for the skewed-normalized/uniform-continuous-ordered-sequential-character-bundle (U-COSCB) approach; model_v0.1.0 for the skewed-normalized/uniform-with-padding-continuous-ordered-sandwich-bundle model (SNWP-COSB). Note that while I was able to get the COSCB variants to work, I could not solve COSB. For explanations of these models, see my paper attached, section "Who Were Eating a Sandwhich: A New Experiment.
4. Each folder has dataprep files, which contain the TextData objects that handle the text processing. We combine this with the masterdataprep file to synthesize training text files into one master file and process that data. We feed this data into the rnn (recurrent neural network) file, which allows us to train the model using Tensorflow. Then we can call the modelpredict file to actually predict based on a trained model for an input textfile. The results are adapted to be published on my Huggingface space as linked above in the files vx.x.x_website.
