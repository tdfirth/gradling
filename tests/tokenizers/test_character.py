import tempfile
from pathlib import Path

from gradling.tokenizers.character import CharacterTokenizer


def test_character_tokenizer():
    tok = CharacterTokenizer.train("abcde")
    assert tok.encode("abcde") == [0, 1, 2, 3, 4]
    assert tok.decode([0, 1, 2, 3, 4]) == "abcde"


def test_character_tokenizer_save_load():
    tok = CharacterTokenizer.train("abcde")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tok.json"
        tok.save(path)
        tok2 = CharacterTokenizer.load(path)
        assert tok.vocab == tok2.vocab
