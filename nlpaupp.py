from textattack.augmentation import WordNetAugmenter 

text = "start each day with positive thoughts and make your day" 
def wnt():
    wordnet_aug = WordNetAugmenter()  # Initialize WordNet augmenter
    wordnet_aug.augment(text)         # Perform augmentation
    for i in range(4):
        print(wordnet_aug.augment(text))
from textattack.augmentation import WordNetAugmenter
def dongnghia():
    augmenter = WordNetAugmenter(pct_words_to_swap=0.1)
    augmented_texts = augmenter.augment("TextAttack is an awesome library for NLP.")
    print(augmented_texts)
from textattack.augmentation import EmbeddingAugmenter

augmenter = EmbeddingAugmenter(pct_words_to_swap=0.1)
augmented_texts = augmenter.augment("TextAttack enhances NLP tasks.")
print(augmented_texts)
