import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/url_lists/all_train.txt"
all_val_urls = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/url_lists/all_val.txt"
all_test_urls = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/url_lists/all_test.txt"

movies_review_tokenized_reviews_dir_train = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/tokened_data/train"
movies_review_tokenized_reviews_dir_gold = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/tokened_data/gold"
finished_files_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/finished_files/gold"
chunks_dir = os.path.join(finished_files_dir, "chunked")
VOCAB_SIZE = 200000
CHUNK_SIZE = 1000

def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    chunk_file(set_name)


def tokenize_reviews(reviews_dir, tokenized_reviews_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  reviews = os.listdir(reviews_dir)
  # make IO list file
  with open("mapping.txt", "w") as f:
    for s in reviews:
      f.write("%s \t %s\n" % (os.path.join(reviews_dir, s), os.path.join(tokenized_reviews_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  subprocess.call(command)
  os.remove("mapping.txt")

  # Check that the tokenized reviews directory contains the same number of files as the original directory
  num_orig = len(os.listdir(reviews_dir))
  num_tokenized = len(os.listdir(tokenized_reviews_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized reviews directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_reviews_dir, num_tokenized, reviews_dir, num_orig))

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def write_to_bin(url_file, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  if makevocab:
    vocab_counter = collections.Counter()

  reviews = os.listdir(movies_review_tokenized_reviews_dir_train)
  with open(out_file, 'wb') as writer:
    for s in reviews:
      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(movies_review_tokenized_reviews_dir_train, s)):
        story_file = os.path.join(movies_review_tokenized_reviews_dir_train, s)

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      tf_example = example_pb2.Example()

      tf_example.features.feature['article'].bytes_list.value.extend([str.encode(article)])
      tf_example.features.feature['abstract'].bytes_list.value.extend([str.encode(abstract)])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

#   print "Finished writing file %s\n" % out_file

  # write vocab to file
  if makevocab:
    # print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')

def check_num_reviews(reviews_dir, num_expected):
  num_reviews = len(os.listdir(reviews_dir))
  if num_reviews != num_expected:
    raise Exception("reviews directory %s contains %i files but should contain %i" % (reviews_dir, num_reviews, num_expected))


def main():
#   if len(sys.argv) != 3:
#     print "USAGE: python make_datafiles.py <movies_review_test_dir> <dailymail_reviews_dir>"
#     sys.exit()
#   movies_review_test_dir = sys.argv[1]
#   dm_reviews_dir = sys.argv[2]
  movies_review_test_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/data/train'
  movies_review_gold_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/data/gold'

  # Create some new directories
  if not os.path.exists(movies_review_tokenized_reviews_dir_train): os.makedirs(movies_review_tokenized_reviews_dir_train)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both reviews dirs, outputting to tokenized reviews directories
  tokenize_reviews(movies_review_test_dir, movies_review_tokenized_reviews_dir_train)
  tokenize_reviews(movies_review_gold_dir, movies_review_tokenized_reviews_dir_gold)

  # Read the tokenized reviews, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

  chunk_all()

main()
