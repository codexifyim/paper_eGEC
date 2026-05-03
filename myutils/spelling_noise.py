import argparse
import csv
import random
import re
from tqdm import tqdm
from multiprocessing import Pool
from filelock import FileLock

def load_word_pairs(csv_file):
    word_pairs = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:  
                correct_word = row[0].strip()
                incorrect_variants = [w.strip() for w in row[1:] if w.strip()]
                word_pairs[correct_word] = incorrect_variants
    return word_pairs

def add_spellingnoise(sentence, word_pairs, error_rate, seed):
    local_random = random.Random(seed)
    if local_random.random() >= error_rate:
        return sentence

    tokens = re.findall(r'\s+|\w+|[^\w\s]', sentence)
    print(f'tokens {tokens}')
    #select a word for spelling noise
    candidate_indices = [i for i, token in enumerate(tokens) if token in word_pairs] 

    if not candidate_indices:
        return sentence 

    selected_index = local_random.choice(candidate_indices)
    tokens[selected_index] = local_random.choice(word_pairs[tokens[selected_index]])

    return ''.join(tokens)

def process_batch(batch, word_pairs, error_rate, seed):
    return [add_spellingnoise(sentence, word_pairs, error_rate, seed + i) for i, sentence in enumerate(batch)]

def write_output(output_file, results):
    lock = FileLock(output_file + ".lock") 
    with lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            # print('writing file')
            f.writelines('\n'.join(results) + '\n')

def main(args):
    word_pairs = load_word_pairs(args.csv_file)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()

    batches = [sentences[i:i + args.batch_size] for i in range(0, len(sentences), args.batch_size)]
    seeds = [args.seed + i * args.batch_size for i in range(len(batches))]  

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("")

    with Pool(processes=args.num_cpus) as pool:
        with tqdm(total=len(batches), desc="processing") as pbar:
            for batch_result in pool.starmap(process_batch, zip(
                batches, [word_pairs] * len(batches), [args.error_rate] * len(batches), seeds
            )):
                write_output(args.output_file, batch_result)
                pbar.update(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o","--output_file", required=True)
    parser.add_argument("-c","--csv_file", required=True)
    parser.add_argument("--error_rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_cpus", type=int, default=1)
    args = parser.parse_args()
    main(args)
