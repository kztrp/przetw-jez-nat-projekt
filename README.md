
# Natural language processing<br/>


## Overview:

The project is carried out as part of an course at the Faculty of Electronics of the [Wrocław University of Technology](http://pwr.edu.pl/en/).

The aim of the project is to acquire practical skills and implementation of natural language processing.
The chosen technique is OCR and the problem is the recognition of units in the content.

## Methods:

We are going to compare the implementations for our dataset of three methods from the NLTK library 


> Methods
>
>  * Unigram Tagger,
>  * Unigram Tagger with Regexp Tagger,
>  * Brill Tagger Trainer.

## Preview Images:


## Prerequisites [for developers]


Install packages with pip: -r requirements.txt

 ```bash
    pip install -r requirements.txt
 ```
Run the Python interpreter and type the commands:

 ```bash
>>> import nltk
>>> nltk.download()
 ```
 
 A new window should open, showing the NLTK Downloader. Click on the File menu and select Change Download Directory. For central installation, set this to C:\nltk_data (Windows), /usr/local/share/nltk_data (Mac), or /usr/share/nltk_data (Unix). Next, select the packages or collections you want to download.

![NLTK]( other/github_images/NLTK.png?raw=true "NLTK Downloader")

## Getting started [for developers]


Install packages with pip: -r requirements.txt

### `$ pip install -r requirements.txt`


## Project Structure

 *Full Breakdown*

 ```sh
 <przetw-jez-nat-projekt-main>
 ├── text_data # Scanned documents for testing
 ├── brill_tagger.py # method brill tagger
 ├── ner.py # Main file. Performs an Named Entity Recognition from the data returned by the tagers.
 ├── README.md # 
 ├── requirements.txt # 
 ├── t_student.py # used for statistical tests
 ├── unigram_tagger.py # method unigram tagger

 ```
 
 A “tag” is a case-sensitive string that specifies some property of a token, such as its part of speech. Tagged tokens are encoded as tuples (tag, token). For example, the following tagged token combines the word 'fly' with a noun part of speech tag ('NN'):
 

## Running the application [for developers]



U can run  `$ python ./t_student.py `  for statistical tests. Run 'ner.py' for performs an Named Entity Recognition.




