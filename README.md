Spelling Checker with a Language Model
======================================

This is my final project for my Natural Language Processing class. It is designed to be an improvement on a basic spelling checker created earlier in the term. These improvements include weighed letter substitution cost, a trigram language model, and a dictionary of known words trained from a corpus.


##Corpora
For my training my trigram language model and dictionary, I used several NLTK corpora. The first was the Gutenberg corpus. This consists of text files for eighteen books and has over two million words. The second corpus used was the Reuters corpus, consisting of ten thousand news documents with around 1.3 million words. The final corpus used in training my model was the Brown corpus. This has excerpts from fifteen different genres and around 1.15 million words.

For testing my spelling checker, I used the Holbrook corpus. This corpus consists of excerpts from the book English for the Rejected. The text was written by "secondary-school children, in their next-to-last year of schooling". This is a corpus of text that has some errors tagged. These included errors in capitalization, conjugation, splitting up or combining words, and errors in spelling. No distinction was made between error type.
