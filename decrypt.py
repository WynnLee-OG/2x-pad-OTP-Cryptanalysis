import subprocess as sp
import os # for file path
import math
import pandas as pd
import numpy as np
import math
import sys
OUTPUT_FILE = 'output/output.txt'
BINARY_SCRATCH_FILE = 'output/bwscratch.txt'
DATABASE_DIRECTORY = 'db'
MAX_BITS_LENGTH = 1024
VALID_CHARACTERS = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class Decrypt:
    def __init__(self):
        self.bigram_prob = {}
        self.unigram_prob = {}
        self.db_path = "/project/web-classes/Fall-2025/csci5471/hw1/db/"
        self.ftable_path = "/project/web-classes/Fall-2025/csci5471/hw1/ftable2.csv"

    def get_cipher(self):
        # get the cipher txt from the CSE pc
        result = sp.run(
            ["/project/web-classes/Fall-2025/csci5471/hw1/make2tp"],
            capture_output=True,
            text=False  # return raw bytes
        )
        cipher_bytes = result.stdout
        if len(cipher_bytes) != 2048:
            # checks if length's correct
            raise ValueError(f"Expected 2048 bytes, got {len(cipher_bytes)}")

        c1 = cipher_bytes[:1024] # ciphertext 1
        c2 = cipher_bytes[1024:] # ciphertext 2
        return c1, c2

    def load_frequencies_from_occurences(self, data_frame): 
        #data_frame = pd.read_csv(self.ftable_path)
        unigram_row = data_frame.iloc[0] # select the first row
        #print("this is unigram_row:", unigram_row)
        total_chars = sum(unigram_row[1:]) # sum of all unigram frequencies

        for col in data_frame.columns[1:]:
            freq = unigram_row[col]
            if freq > 0:
                probab = freq / total_chars
                self.unigram_prob[col] = math.log(probab)
            else:
                self.unigram_prob[col] = -1e9
        
        for index, row in data_frame.iloc[1:].iterrows():
            first_letter = row['first']
            total_bigrams = sum(row[1:]) # sum of all bigram frequencies for first letter
            for col in data_frame.columns[1:]:
                next_char = col
                freq = row[col]
                if freq > 0:
                    probab = freq / total_bigrams
                    self.bigram_prob[(first_letter, next_char)] = math.log(probab)
                else:
                    self.bigram_prob[(first_letter, next_char)] = -1e9

    # The function below could be useful if we want to think of a better approach to this problem.
    def generate_histogram_from_text(self, text:str):
        histogram = pd.DataFrame()
        endpoint = len(text) - 1
        for index, character in enumerate(text):
            if index < endpoint: # stop before the last one, so we don't go past the end of the array
                histogram[0][character] += 1 #increment occurences of this character
                histogram[character][text[index+1]] += 1 # increment occurences of the bigram of this character plus the next character

        histogram[text[-1]][0] += 1 # make sure we account for the last character when computing unigram efficiency


        return histogram
                


    def load_frequencies(self): # reads csv file using panda 
        data_frame = pd.read_csv(self.ftable_path)
        unigram_row = data_frame.iloc[0] # select the first row
        #print("this is unigram_row:", unigram_row)
        total_chars = sum(unigram_row[1:]) # sum of all unigram frequencies

        for col in data_frame.columns[1:]:
            freq = unigram_row[col]
            if freq > 0:
                probab = freq / total_chars
                self.unigram_prob[col] = math.log(probab)
            else:
                self.unigram_prob[col] = -1e9
        
        for index, row in data_frame.iloc[1:].iterrows():
            first_letter = row['first']
            total_bigrams = sum(row[1:]) # sum of all bigram frequencies for first letter
            for col in data_frame.columns[1:]:
                next_char = col
                freq = row[col]
                if freq > 0:
                    probab = freq / total_bigrams
                    self.bigram_prob[(first_letter, next_char)] = math.log(probab)
                else:
                    self.bigram_prob[(first_letter, next_char)] = -1e9

    def chance_english(self, int_text):
        '''tests the probability that this is an english document by multiplying all unigram frequencies and bigram frequencies of all occuring grams together. Does not work if files can be different lengths. Rather than dealing with underflow errors, uses logs instead of multiplying probabilities.'''
        '''Assumes that there are no special characters, since we don't have bigram frequencies for them'''
        text = self.int_to_string(int_text)
        prob_log = 0
        endpoint = len(text) - 1

        for index, character in enumerate(text):
            if (index < endpoint):
                next_character = text[index + 1]
                
                if character not in VALID_CHARACTERS:
                    #if index != 0:
                    #    print("THIS", character, index)
                    prob_log = -math.inf
                    break
                elif next_character not in VALID_CHARACTERS:
                    
                    #if index != 0:
                    #    print("NEXT", next_character, index)
                    prob_log = -math.inf
                    break
                
                else: #stop one before the end
                    
                    # first multiply the probability that this character were to occur in a vacuum using the unigram probability
                    prob_log += self.unigram_prob[character]
                    # then, do the same thing with the bigram probability (this character, next character)
                    prob_log += self.bigram_prob[(character, next_character)]


        if prob_log != -math.inf:
            print("PROBABILITY:", prob_log)
        return prob_log

    def string_to_int(self, string):
        # Concept stolen from stackoverflow'
        return int.from_bytes(string.encode('utf-8'), byteorder=sys.byteorder)
    def int_to_string(self, b):
        return b.to_bytes(math.ceil((b).bit_length() / 8), byteorder = sys.byteorder).decode('utf-8')

    '''def XOR_bits(b1, b2, byteorder=sys.byteorder) -> int:
        # This concept also was stolen from stack overflow
        key, var = key[:len(var)], var[:len(key)] #shorten them to make sure they are the same length
        int_var = int.from_bytes(var, byteorder) # Convert to integer
        int_key = int.from_bytes(key, byteorder)
        int_enc = int_var ^ int_key #bitwise xor the integers
        return int_enc
'''


    def char_from_byte(self, byte): # 65-90-> A-Z, 97->122->a-z, 32->space
        if byte == 32:
            return ' '
        elif 65 <= byte <= 90:
            return chr(byte)
        elif 97 <= byte <= 122:
            return chr(byte - 32) # convert to uppercase
        else:
            return ' '

    def score_text(self, text_bytes):
        text_string = ""

        for byte in text_bytes:
            text_string += self.char_from_byte(byte)
        score = 0.0

        for i in range(len(text_string) - 1):
            bigram = (text_string[i], text_string[i+1])
            if bigram in self.bigram_prob:
                score += self.bigram_prob[bigram]
            else:
                if text_string[i+1] in self.unigram_prob:
                    score += self.unigram_prob[text_string[i + 1]] - 10.0
                else:
                    score -= 100.0
        return score

    def load_db_files(self):
        db_files = []
        for filename in os.listdir(self.db_path):
            filepath = os.path.join(self.db_path, filename)
            with open(filepath, 'rb') as file:
                content = file.read(1024)
                if len(content) == 1024:
                    db_files.append(content)
        return db_files

    def decrypt(self):
        c1, c2 = self.get_cipher()
        x = bytes(a^b for a, b in zip(c1, c2)) # XOR of c1 and c22
        db_files = self.load_db_files()
        self.load_frequencies()
        best_score = float('-inf')
        best_f = None
        best_e = None

        for f_candidate in db_files:
            e_candidate = bytes(a^b for a, b in zip(x, f_candidate))
            score = self.score_text(e_candidate)
            if score > best_score:
                best_score = score
                best_f = f_candidate
                best_e = e_candidate
        
        with open("decrypted_database.bin", "wb") as file:
            file.write(best_f)
        with open("decrypted_english.txt", "wb") as file:
            file.write(best_e)
        print("Decryption completed")

if __name__ == "__main__":

    dc = Decrypt()
    dc.decrypt()