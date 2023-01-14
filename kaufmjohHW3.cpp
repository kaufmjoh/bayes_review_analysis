/* John Kaufman
 * email: kaufmjoh@oregonstate.edu
 *
 * Spring 2020 CS 331 Programming Assignment 3
 *
 * Use Naive Bayes nets to classify a set of restaurant reviews as positive or negative, based on a training set
 *
 * */

/*
 * This file was not was submitted and graded. That file was terrible, a result of me writing it after procastinating and being sleep-deprived.
 *
 * I went back and wrote this current iteration of the file before receiving a grade on what I submitted. I am happier with this file
 * This file still fails to use log states.
 * */

#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;

struct dynarray
{
	string* words;
	int size;
	int capacity;
};

int number_of_sentences_in_set(char* path);
int word_exists_in_sentence(string word, struct dynarray*);
void normalize_sentence(struct dynarray*, string);

float count_instances(int type, int word_index, int** features, int num_sentences, int size);

void process_data(int**, char* path, int num_sentences, int, struct dynarray* vocab);
void write_to_file(int output, struct dynarray* vocab, int num_sentences, int** features);

void alphabetize(struct dynarray* vocab);
void enqueue(struct dynarray* vocab, string word);
void get_vocab(struct dynarray* vocab, char* path);
bool normalize(string& word, string& extra_word);
string remove_null(string word);
void display_vocab(struct dynarray* vocab);


//When running this program, use command line arguments as follows: executableName, trainingSetName, testSetName

int main(int argc, char* argv[])
{
//BEGIN PRE-PROCESSING

	struct dynarray* vocab = new struct dynarray;
	vocab->size = 0;
	vocab->capacity = 2;
	vocab->words = new string[2];

	
	//get the vocab from the training set, and then alphabetize it
	get_vocab(vocab, argv[1]);
	alphabetize(vocab);

	//count and store the number of sentences in the training set and the testing set
	int num_training_sentences = number_of_sentences_in_set(argv[1]);
	int num_test_sentences = number_of_sentences_in_set(argv[2]);

	//Create a set of features to store the processed data for the training and test set
	int** train_features = new int*[num_training_sentences];
	for(int i = 0; i < num_training_sentences; i++)
		train_features[i] = new int[vocab->size+1];

	int** test_features = new int*[num_test_sentences];
	for(int i = 0; i < num_test_sentences; i++)
		test_features[i] = new int[vocab->size+1];

	//Process the data for training and test set, storing the results in their respective features
	process_data(train_features, argv[1], num_training_sentences, 0, vocab);
	process_data(test_features, argv[2], num_test_sentences, 1, vocab);

//END OF PRE-PROCESSING

/**************************************************************************************************/	

//BEGIN CLASSIFICATION
	
	//Calculate the percentage of reviews in the training set that are positive
	float num_pos = 0;
	float num_neg = 0;
	float prob_pos;

	for(int i = 0; i < num_training_sentences; i++)
	{
		if(train_features[i][vocab->size] == 0)
			num_neg++;
		else
			num_pos++;
	}

	prob_pos = (num_pos) / (num_pos + num_neg);

	//for each word, count the number of times:
	//	0: the word is absent and the review is negative
	//	1: the word is present and the review is negative
	//	2: the word is absent and the review is positive
	//	3: the word is present and the review is positive
	float count_records[4][vocab->size];
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < vocab->size; j++)
		{
			count_records[i][j] = count_instances(i, j, train_features, num_training_sentences, vocab->size);
		}
	}

	//For each word, calculate the probability of P(word | label) using uniform dirichlet priors
	float prob_given_label[4][vocab->size];
	for(int i = 0; i < vocab->size; i++)
	{
		prob_given_label[0][i] = (count_records[0][i]+1) / (num_neg + 2); //word is 0, label is 0
		prob_given_label[1][i] = (count_records[1][i]+1) / (num_neg + 2); //word is 1, label is 0
		prob_given_label[2][i] = (count_records[2][i]+1) / (num_pos + 2); //word is 0, label is 1
		prob_given_label[3][i] = (count_records[3][i]+1) / (num_pos + 2); //word is 1, label is 1

	}	


	float test_guess[num_test_sentences];
	float pos_guess = prob_pos;
	float neg_guess = (1-prob_pos);
	float confidence;

	//Calculate the probability the review is positive or negative.
	//If there is a higher probability of the review being positive, store a 1, otherwise stroe 0
	for(int i = 0; i < num_test_sentences; i++)
	{
		pos_guess = prob_pos;
		neg_guess = (1-prob_pos);

		//For each word, take the current probability the review is postive or negative
		//And multiply it by the probability a word would exist in a positive or negative review
		for(int j = 0; j < vocab->size; j++)
		{
			//Update the probability the review is positive	
			if(test_features[i][j] == 0)
				pos_guess = pos_guess * prob_given_label[2][j];
			else
				pos_guess = pos_guess * prob_given_label[3][j];
		
			//Update the probability the review is negative
			if(test_features[i][j] == 0)
				neg_guess = neg_guess * prob_given_label[0][j];
			else
				neg_guess = neg_guess * prob_given_label[1][j];
		}	

		//If there is a higher chance the review is positive, store a 1, otherwise store a 0
		if(pos_guess > neg_guess)
		{
			test_guess[i] = 1;
			confidence = pos_guess / neg_guess;
		}
		else
		{
			test_guess[i] = 0;
			confidence = neg_guess / pos_guess;
		}
		if(num_test_sentences < 100)
		{
			cout << "\nSentence " << i << " is " << test_guess[i] << ", which is " << confidence << " times as likely as the alternative" << endl;

			cout << "\tPos_guess is: " << pos_guess << "\tNeg_guess is: " << neg_guess << endl;
		}
		//	cout << pos_guess << "\t" << neg_guess << "\t" << test_guess[i] << endl;
		//	cout << test_features[i][vocab->size-1] << endl;
		//	cin.get();
	}

	//Count the number of correct predictions, and report the accuracy
	int num_correct = 0;
	float percent_correct = 0;
	for(int i = 0; i < num_test_sentences; i++)
	{
		if(test_guess[i] == test_features[i][vocab->size])
			num_correct++;
	}

	percent_correct = (float)num_correct / (float)num_test_sentences;
	cout << "The accuraccy of the predictions is: " << num_correct << "/" << num_test_sentences << " = "<< percent_correct << endl;

	//Write results to results.txt
	ofstream out;
	out.open("results.txt");
	for(int i = 0; i < num_test_sentences; i++)
		out << test_guess[i] << ",";
	out << "\n\nThe training file used was: " << argv[1] << " and the testing file used was: " << argv[2] << endl;
	out << "\nThe accuraccy of the predictions is: " << num_correct << "/" << num_test_sentences << " = "<< percent_correct << endl;
	out.close();

	//END OF CLASSIFICATION

}

/*float count_instances(int type, int word_index, int** features, int num_sentences, int size)
 * 
 * Given a set of processed data in features, count the number of times something happens based on type
 * If type is
 * 	0: count the instances where a word does not exist and the review was negative
 * 	1: count the instances where a word exists and the review was negative
 * 	2: count the instances where a word does not exist and the review was postive
 * 	3: count the instances where a word exists and the review was positive
 *
 * return the number of times that thing was counted
 * 
 */
float count_instances(int type, int word_index, int** features, int num_sentences, int size)
{
	float counter = 0;


	for(int i = 0; i < num_sentences; i++)
	{
		if(type == 0 && features[i][word_index] == 0 && features[i][size] == 0)
		{
			counter++;
		}
		if(type == 1 && features[i][word_index] == 1 && features[i][size] == 0)
		{
			counter++;
		}
		if(type == 2 && features[i][word_index] == 0 && features[i][size] == 1)
		{	
			counter++;
		}
		if(type == 3 && features[i][word_index] == 1 && features[i][size] == 1)
		{
			counter++;
		}
	}

	return counter;
}

/* void process_data(int** features, char* path, int num_sentences, int output, struct dynarray* vocab)
 *
 * for every setence in a sentence set specified in path, check to see if that sentence contains each
 * of the words in vocab. If yes, then store a 1 in the corresponding features slot. If not, store 0.
 *
 * Once the features have been processed, write the processed data to a text file
 */
void process_data(int** features, char* path, int num_sentences, int output, struct dynarray* vocab)
{
	ifstream in;
	in.open(path);
	int label;

	//for each sentence
	for(int i = 0; i < num_sentences; i++)
	{
		string line;
		//read in a line, and store the class label
		getline(in, line);

		label = line.at(line.length()-3);

		label = label - 48;

		//create a new sentence struct
		struct dynarray* sentence = new struct dynarray;
		sentence->size = 0;
		sentence->capacity = 2;
		sentence->words = new string[2];

		//normalize the sentence
		normalize_sentence(sentence, line);

		//check each word of the vocabulary to see if it exists in the sentence, 1 if yes, 0 if no
		for(int j = 0; j < vocab->size; j++)
		{
			features[i][j] = word_exists_in_sentence(vocab->words[j], sentence);
		}

		//store the label in the features
		//features[i][vocab->size-1] = -999999;
		features[i][vocab->size] = label;
	}

	in.close();

	//write the processed data to a text file
	write_to_file(output, vocab, num_sentences, features);	

}

//void write_to_file(int output, struct dynarray* vocab, int num_sentences, int** features)
//
//open a file based on the specified output, and write the processed data provided in features
//to the file. 
void write_to_file(int output, struct dynarray* vocab, int num_sentences, int** features)
{
	ofstream out;
	//open the correct file
	if(output == 0)
		out.open("preprocessed_train.txt");
	else
		out.open("preprocessed_test.txt");

	//write the vocab
	for(int i = 1; i < vocab->size; i++)
		out << vocab->words[i] << ",";

	//write the last non-word, classlabel
	out << "classlabel" << endl;

	//write teh processed data to the file
	for(int i = 0; i < num_sentences; i++)
	{
		for(int j = 0; j < vocab->size; j++)
		{
			out << features[i][j];
			if(j + 1 != vocab->size)
				out << ",";
		}
		out << endl;
	}

	out.close();
}

//Given a dynamic array of strings, and a string word to search for,
//return 1 if the word exists in the sentence, and 0 if it does not
int word_exists_in_sentence(string word, struct dynarray* sentence)
{

	//check every word of sentence looking for match
	for(int i = 0; i < sentence->size; i++)
	{
		if(sentence->words[i] == word)
		{
			return 1;
		}
	}

	return 0;
}

/*void normalize_sentence(struct dynarray* sentence, string line)
 *
 * given a line of text with possibley many words,
 * normalize the words and store them in the sentence dynamic array
 *
 */
void normalize_sentence(struct dynarray* sentence, string line)
{
	bool flag;

	//replace all whitespaces with periods, so the normalize function works
	for(int i = 0; i < line.length(); i++)
	{
		if(line.at(i) == ' ')
			line.at(i) = '.';
	}

	string word;
	string extra_word = "Blank\0";

	//normalize, storing the first valid word in line
	flag = normalize(line, extra_word);

	//add the word to the sentence
	enqueue(sentence, line);

	//while there are additional words behind periods, keep extracting them and adding them to the sentence
	while(flag)
	{
		word = extra_word;
		extra_word = "Blank\0";

		flag = normalize(word, extra_word);

		enqueue(sentence, word);
	}

}

/* void alphabetize(struct dynarray* vocab)
 *
 * Given a vocabulary that is an array of strings, alphabetize them
 *
 */
void alphabetize(struct dynarray* vocab)
{
	string temp;

	for(int i = 0; i < (vocab->size-1); i++)
	{
		for(int j = 0; j < ((vocab->size)-i-1); j++)
		{
			if(vocab->words[j] > vocab->words[j+1])
			{
				temp = vocab->words[j];
				vocab->words[j] = vocab->words[j+1];
				vocab->words[j+1] = temp;
			}
		}
	}
}


/* void enqueue(struct dynarray* vocab, string word)
 *
 * Add a word to the vocabulary
 *
 * the vocabulary is a dynamic array of strings
 * if the array is full, resize and copy all old elements
 *
 * put the new word in the array
 *
 */
void enqueue(struct dynarray* vocab, string word)
{
	bool new_word = true;

	//check to see if the word already exists. if it does, it is not added
	for(int i = 0; i < vocab->size; i++)
		if(vocab->words[i] == word)
			new_word = false;


	//if it is a new word, add it 
	if(new_word)
	{
		//if the array is full, create one twice as large, and copy all old elements
		if(vocab->size == vocab->capacity)
		{

			string* resized = new string[2*(vocab->capacity)];

			for(int i = 0; i < (vocab->size); i++)
				resized[i] = (vocab->words)[i];

			vocab->words = resized;

			vocab->capacity = 2*(vocab->capacity);
		}

		//add the new word to the vocab
		vocab->words[vocab->size] = word;
		vocab->size++;

	}
}


/*
 * void get_vocab(struct dynarray* vocab, char* path)
 *
 * Create a vocabulary from a set of sentences.
 * Store the vocabulars in a dynamic array of strings, vocab.
 *
 * The sentences come from the file passed as path
 */
void get_vocab(struct dynarray* vocab, char* path)
{
	int num_resulting_words;

	ifstream in;
	in.open(path);

	string word;
	string extra_word = "Blank\0";

	bool extras;


	//for the whole file
	while(!in.eof())
	{
		extra_word.at(0) = '\0';

		//read in some text separated by a space
		in >> word;

		//normalize the text
		extras = normalize(word, extra_word);

		//store the normalized word in the vocab
		enqueue(vocab, word);

		//while there are hidden words, extract them, and add to vocab
		while(extras)
		{
			word = extra_word;
			extra_word = "Blank\0";
			extras = normalize(word, extra_word);
			enqueue(vocab, word);
		}

	}

	in.close();
}


/* bool normalize(string& word, string& extra_word)
 *
 * given a string of text in word, normalize it, (remove everything that's not a letter, send all letters to lowercase)
 * 
 * if there is a suspected hidden extra_word (validtext...moretext), 'validtext' is stored in word, and 'moretext' is stored in extra word
 * 	and the function returns true to indicate there is a hidden word
 *
 * if there is no hidden word, the function returns false
 */
bool normalize(string& word, string& extra_word)
{

	bool hidden_word = false;
	int first_null = 0;

	//remove everything that's not a letter, a period, a dash, or a comma
	for(int i = (word.length() - 1); i >= 0; i--)
	{
		if((word.at(i) < 65 && word.at(i) != 44 && word.at(i)!= 45 && word.at(i) != 46) || (word.at(i) > 90 && word.at(i) < 97) || word.at(i) > 122)
		{

			for(int j = i; j < (word.length()-1); j++)
			{
				word.at(j) = word.at(j+1);
			}
			word.at(word.length()-1) = '@';

		}
	}	

	//turn all uppercase letters into lowercase letters
	for(int i = 0; i < word.length(); i++)
	{
		if(word.at(i) > 64 && word.at(i) < 91)
			word.at(i) += 32;
	}

	word = remove_null(word);
	//if there is a period or comma followed by a letter, denote that it's most likely word...word
	//otherwise, strip away periods and commas
	for(int i = 0; i < (word.length()); i++)
	{
		if(word.at(i) == 44 || word.at(i) == 45|| word.at(i) == 46)
		{
			for(int j = i; j < (word.length()-1);j++)
			{
				//if there is valid text after a found period, comma, dash
				if(word.at(j) != 44 && word.at(j) != 45 && word.at(j) != 46)
				{
					hidden_word = true;

					while(word.at(j) == 44 || word.at(j) == 45 || word.at(j) == 46)
						j++;

					//store the hidden word in extra_word
					extra_word = word.substr(j, word.length());

					for(i; i < (word.length()); i++)
						word.at(i) = '@';

					//store the first part of text in word
					word = remove_null(word);
					return true;
				}
			} 

			//if there is no more valid text after period, put nothing in extra_word
			if(!hidden_word)
			{
				for(i; i < (word.length()); i++)
					word.at(i) = '@';
				word = remove_null(word);
				return false;	
			}
		}
	}
	return false;

}

//Helper function for the normalize funtion:
//When an illegal character is found, it is replaced with '@'.
//This function removes all the '@'s in a string
string remove_null(string word)
{

	int first_null = 0;
	while(first_null < word.length() && word.at(first_null)!='@')
		first_null++;
	return word.substr(0, first_null);
}


//Auxillary function: count the number of setences in a set
int number_of_sentences_in_set(char* path)
{
	ifstream in;
	in.open(path);
	int num = 0;
	string word;

	while(getline(in, word))
	{
		num++;
	}

	return num;
}


//Auxiallary function: Display all the words in the vocab. Commented block gives the option to view all ASCII chars of selected word
void display_vocab(struct dynarray* vocab)
{

	for(int i = 0; i < vocab->size; i++)
	{
		cout << "Word: " << i << ": <" << vocab->words[i] << ">. Length of word: " << vocab->words[i].length() << endl;
	}

	/*	int a,b;
		cout << "Enter the index of a word to see details of: " << endl;
		cin >> a;
		cout << "Enter the index of a word to see details of: " << endl;
		cin >> b;

		for(int i = 0; i < vocab->words[a].length(); i++)
		cout << (int)vocab->words[a].at(i) << " ";
		cout << endl;
		for(int i = 0; i < vocab->words[b].length(); i++)
		cout << (int)vocab->words[b].at(i) << " ";
		cout << endl;*/
}
