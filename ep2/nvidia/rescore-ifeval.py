import os
from pathlib import Path
import runpy
import sys

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from nltk.tabdata import PunktDecoder
import nltk.tokenize
from evalscope.benchmarks.ifeval import instructions_util

directory=Path(os.environ["NLTK_DATA"])/"tokenizers/punkt_tab/english"
decoder=PunktDecoder()
params=PunktParameters()
for name, field, decode in (
    ("collocations.tab","collocations",decoder.tab2tups),
    ("sent_starters.txt","sent_starters",decoder.txt2set),
    ("abbrev_types.txt","abbrev_types",decoder.txt2set),
    ("ortho_context.tab","ortho_context",decoder.tab2intdict),
):
    with (directory/name).open(encoding="utf-8") as source:
        setattr(params,field,decode(source))
tokenizer=PunktSentenceTokenizer(params)
assert tokenizer.tokenize("Hello there. Goodbye now.")==["Hello there.","Goodbye now."]
instructions_util._get_sentence_tokenizer=lambda: tokenizer
original_punkt_loader=nltk.tokenize._get_punkt_tokenizer
nltk.tokenize._get_punkt_tokenizer=lambda language: tokenizer if language=="english" else original_punkt_loader(language)
assert nltk.tokenize.sent_tokenize("Hello there. Goodbye now.")==["Hello there.","Goodbye now."]
assert nltk.tokenize.word_tokenize("Hello there.")==["Hello","there","."]
assert instructions_util.count_sentences("Hello there. Goodbye now.")==2
print("Verified English Punkt sentence tokenizer for IFEval",flush=True)
runpy.run_path(str(Path(sys.executable).with_name("evalscope")),run_name="__main__")
