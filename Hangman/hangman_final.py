#
# Hangman game that allows for one consistent lie.
# All words must exist in dictionary.txt file, downloaded from
#   https://raw.githubusercontent.com/raypereda/hangman/master/words.txt
#
# See paper for more details and rules.
#
# Author: Aditya Singhvi
# Part of group project w/ Arya Maheshwari
# Information Theory 1, Fall 2020, The Harker School
#
# January 12, 2021

#
# CONSTANTS ------------------------------------------------------
#
fh = open("words.txt")
dictionary = [x.strip() for x in fh.readlines()]

a_offset = 97  # ASCII value of lowercase a


#
# HELPER METHODS -----------------------------------------------------------
#

#
# Generates a probability distribution of letters for a given word list
# Given a list of words, generates a 26-value array that describes the likelihood
# of the corresponding letter (0-->A, 1-->B,...25-->Z) appearing in a randomly
# selected word from the list.
#
# w_list: iterable containing the word-strings for which a distribution is to be generated
#
def prob_dist(w_list):
    letters = []
    for j in range(0, 26):
        letters.append(0)

    for string in w_list:
        for j in range(a_offset, a_offset + 26):
            if chr(j) in string:
                letters[j - 97] += 1

    total = float(len(w_list))

    for j in range(0, 26):
        letters[j] /= total

    return letters


#
# Given a probability distribution letters_dist and an iterable with the known
# information about a word, cleans up the probability distribution such that the
# probability of any letter known to be in the word is set to 0 and returns it.
#
# PARAMS:
#   letters_dist: len 26 array of floats containing probabilities for each letter
#                   in order from a to z
#   known: string containing known information about a word
#           alphabetic characters in positions where the character is known
#           non-alphabetic characters otherwise
#           For instance, the word "hippopotamus" could be represented at some point
#           as *ipp*p**a***. Probabilities corresponding to i, p, a would be set to 0.
#
# Returns: Cleaned up probability distribution.
#
def smart_prob_dist(letters_dist, known):
    for l in known:
        if l.isalpha():
            letters_dist[ord(l) - a_offset] = 0
    return letters_dist


#
# Given a word, a string representing the known letters in the word,
# and another iterable representing the eliminated characters, checks
# whether the word is "valid" and returns either True or False.
#
# A word is considered valid if all alphabetic characters in word known
# appear in the same positions in the word. Furthermore, none of the
# eliminated characters may appear in the word.
#
# Params:
#   word: the word to be checked for validity (string)
#   known: string representing information known abt word so far.
#          For instance, the word "hippopotamus" could be represented at some point as *ipp*p**a***.
#   eliminated: iterable sequence of chars representing letters that may not be present in word
#
def is_word_valid(word, known, eliminated):
    for ch in eliminated:
        if ch in word:
            return False

    for j in range(0, len(known)):
        if known[j].isalpha() and known[j] != word[j]:
            return False

    return True


#
# Given a list of words, returns a sub-list of valid words given the known word-string and
# eliminated letters
#
def eliminate_words(existing_list, known, eliminated):
    return [word for word in existing_list if is_word_valid(word, known, eliminated)]


#
# Generates a list of words of a specified length from the pre-set dictionary.
#
def gen_init_word_list(length):
    return [word for word in dictionary if len(word) == length]


#
# Used to group together operations for a single word in the phrase
#
class Word:
    length = 0      # length of the word
    init_word_list = []     # the initial, dictionary-based word-list

    eliminated_letters = []  # letters that have been eliminated by user—could include a lie unless lied = True

    known = []           # the known letters of the word

    possible_word_lists = []  # list containing lists of possible words

    guessed = False     # set when word completely guessed
    lied = False      # set when lie detected
    unknown_word = False

    # Initializes class variables
    def __init__(self, word_length):
        self.length = word_length
        self.init_word_list = gen_init_word_list(word_length)
        self.possible_word_lists = [self.init_word_list]
        self.eliminated_letters = []
        self.known = ["*"] * word_length
        self.guessed = False
        self.lied = False

    # String representation — just the known letters
    def __repr__(self):
        return "".join(self.known)

    # Updates "known" array of letters
    # * in every unknown position
    # correct letter in every known position
    #
    # If no * left after updating, sets self.guessed flag to true
    def update_known(self, new_known):
        for i in range(0, self.length):
            if len(new_known) > i and self.known[i] == "*" and new_known[i].isalpha():
                self.known[i] = new_known[i]

        if "*" not in self.known:
            self.guessed = True

    # Given the list of letters that were wrong guesses (size n), creates the possible
    # n + 1 word lists if no lie has been detected yet. Each wrong letter could be a lie,
    # or none of them could be a lie.
    #
    # If lie already detected, than assumes all wrong letters genuinely wrong.
    def update_word_lists(self, wrong_letters):
        self.possible_word_lists = [eliminate_words(self.init_word_list, self.known, self.eliminated_letters)]
        if not self.lied:
            for i in range(0, len(wrong_letters)):
                temp_elim = self.eliminated_letters.copy()
                temp_elim.remove(wrong_letters[i])
                self.possible_word_lists += eliminate_words(self.init_word_list, self.known, temp_elim)

    # Based on the possible word lists, returns a normalized probability distribution for each letter
    # of size 26.
    #
    # First calculates the probability distribution for each potential word list using methods
    # prob_dist and smart_prob_dist, setting all known letters to 0. Each probability distribution
    # is weighted by the number of words in that potential word list.
    #
    # Adds all these distributions together, then divide by the combined length of every potential word
    # list to create a normalized probability distribution that is then returned.
    #
    # If no potential words remain, sets self.unknown_word to True and simply begins guessing most likely
    # letters based on the length of the word.
    #
    def return_dist(self, wrong_letters):
        expected_dist = [0.0] * 26

        for word_list in self.possible_word_lists:
            if word_list:  # ensure list is not empty
                for i in range(0, 26):
                    expected_dist[i] += smart_prob_dist(prob_dist(word_list), self.known)[i] * len(word_list)

        if not self.unknown_word:
            try:
                # this  line sums the lengths of each word list and then divides all probabilities by that number
                return [prob/sum([len(word_list) for word_list in self.possible_word_lists]) for prob in expected_dist]
            except ZeroDivisionError:
                print("The word with known information", self, "could not be found in the dictionary.")
                print("Weird things may now occur. Nothing is guaranteed. This code has officially taken an L.")
                self.unknown_word = True

        if self.unknown_word:
            expected_dist = smart_prob_dist(prob_dist(self.init_word_list), self.known)
            print(self.eliminated_letters)
            for c in self.eliminated_letters:
                if self.lied or (c not in wrong_letters):
                    expected_dist[ord(c) - a_offset] = 0

        return expected_dist

    # Returns the word if it is known; otherwise returns None.
    def best_guess(self):
        if self.guessed:
            return "".join(self.known)
        return None

    # Called by the game once a lie is detected to set the lie flag to True
    # and update the eliminated letters list.
    def set_lied(self, letter):
        self.lied = True
        self.eliminated_letters.remove(letter)


# -----------------------------------------------------------------------------
# play_game() is essentially the main method, containing the game loop.
#
# Begins by asking for comma-separated list of word lengths, and creates an
# array of Word objects with the specified lengths.
#
# Stores game history in two arrays: guessed_letters and wrong_letters.
# Guessed letters represent all letters that have been guessed so far.
# Wrong letters represent the guessed letters that have not been in any of the words.
# Once a lie is detected and confirmed, that letter is removed from the Wrong letters array.
#
# Game loop then begins and continues until all words have been fully guessed.
#   First, the probability distribution for each Word is retrieved using the return_dist method in
#      the word class.
#   With each letter, the final probability distribution = 1 - (1 - p1)*(1-p2)*(1-p3), essentially
#       yielding the probability that the letter appears in at least one of the words in the phrase.
#   The letter with the maximum probability is then guessed. This makes sense because you want to
#       minimize the number of wrong guesses, not necessarily ask the letter with the greatest entropy.
#   The user then inputs the relevant positions of the character in each word of the phrase, which
#
#
def play_game():
    word_lens = input("Hi! Welcome to Hangman. Please enter a comma-separated list of word lengths:\n")

    words = []  # array of the Word objects used in this game
    for i in word_lens.split(","):
        words.append(Word(int(i)))

    guessed = False         # Flag to decide when to exit game loop
    guessed_letters = []    # Letters that have been guessed by the computer already
    wrong_letters = []      # Letters that were in NONE of the words in the phrase, representing a wrong guess

    num_wrong = 0           # How many fully-wrong guesses

    while not guessed:  # BEGIN GAME LOOP
        guessed = True

        # the overall probability distribution based on which guess is made
        overall_prob_dist = []
        for j in range(0, 26):
            overall_prob_dist.append(1)

        print("Phrase so far: ")
        for word in words:
            print(word)
            word.update_word_lists(wrong_letters)
            word_dist = word.return_dist(wrong_letters)
            for j in range(0, 26):
                overall_prob_dist[j] *= 1 - word_dist[j]

        max_pos = 0
        max_prob = 0

        for j in range(0, 26):
            val = 1 - overall_prob_dist[j]
            overall_prob_dist[j] = val
            if val > max_prob:
                max_prob = val
                max_pos = j

        # guesses the letter most likely to be in at least one word in phrase
        my_guess = chr(max_pos + a_offset)
        guessed_letters += my_guess

        print("My guess is ", my_guess)

        wrong = True    # wrong flag if all letters wrong

        for i in range(0, len(words)):  # iterate over words
            ans = input("Type out word %d with * for all letters that are not %s :\n" % (i + 1, my_guess))

            if not ans.islower():
                words[i].eliminated_letters += my_guess
            else:
                wrong = False
                words[i].update_known(list(ans))

            # CHANGE THIS WIN CONDITION
            if words[i].best_guess() is None:
                guessed = False

        if wrong:
            if my_guess in wrong_letters:
                # wrong letters essentially stores potential lies. once a letter has been
                # confirmed wrong twice, no reason to keep storing it in wrong letters.
                # this is, in fact, badly-written code (at the very least, subpar variable naming).
                # Yet, it works! And as of 1:21 a.m. on this day, I do not want to modify it.
                wrong_letters.remove(my_guess)
            else:
                wrong_letters += my_guess
            num_wrong += 1
        else:       # LIE DETECTOR
            if my_guess in wrong_letters:   # checks if the thing that was not wrong was earlier wrong
                print("You lied! I am disappointed.")
                for word in words:
                    word.set_lied(my_guess)
                num_wrong -= 1
    # END GAME LOOP -----------------------

    if guessed:
        print("My guess is:")
        for word in words:
            print(word.best_guess())
        print("HAHA I WON! It took me %d wrong guesses" % num_wrong)
    else:
        print("Oops I lost.")
# END PLAY GAME

play_game()
