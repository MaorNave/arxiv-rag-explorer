import json

import pytest

from rag_app.dataset import (
    DatasetError,
    JsonlAbstractLoader,
    arxiv_url,
    file_sha256,
    resolve_dataset_path,
    validate_jsonl_sample,
)

from .conftest import RECORDS, write_jsonl


def test_loader_streams_abstracts_and_keeps_only_id_and_title(dataset):
    loader = JsonlAbstractLoader(dataset)
    docs = list(loader.lazy_load())

    # 5 valid records + the duplicate id (dedup happens at indexing time)
    assert len(docs) == 6
    first = docs[0]
    assert first.page_content.startswith("We propose RLBFF")
    assert first.metadata == {"doc_id": "2509.00001v1", "title": RECORDS[0]["title"]}  # authors etc. dropped
    assert loader.stats.malformed == 1
    assert loader.stats.missing_text == 1
    assert loader.stats.lines == 8


def test_loader_normalizes_whitespace_bom_and_field_aliases(tmp_path):
    path = tmp_path / "alias.jsonl"
    path.write_bytes(
        "﻿".encode() + json.dumps({"paper_id": 42, "name": "A  title", "text": "line one\n  line two"}).encode()
    )
    [doc] = JsonlAbstractLoader(path).lazy_load()
    assert doc.page_content == "line one line two"
    assert doc.metadata == {"doc_id": "42", "title": "A title"}


def test_resolve_dataset_path_variants(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    real = write_jsonl(data / "arxiv_2.9k.jsonl", RECORDS)

    assert resolve_dataset_path(str(real)) == real.resolve()
    assert resolve_dataset_path(str(data)) == real.resolve()  # directory -> the .jsonl inside
    # DATA_PATH points to a file that does not exist (arxiv_5k vs arxiv_2.9k): use the sibling.
    assert resolve_dataset_path(str(data / "arxiv_5k.jsonl")) == real.resolve()

    with pytest.raises(DatasetError):
        resolve_dataset_path(str(tmp_path / "nowhere" / "x.jsonl"))


def test_fingerprint_changes_with_content(tmp_path):
    path = write_jsonl(tmp_path / "a.jsonl", RECORDS[:2])
    before = file_sha256(path)
    write_jsonl(path, RECORDS[:3])
    assert file_sha256(path) != before


def test_validate_jsonl_sample(tmp_path):
    validate_jsonl_sample(write_jsonl(tmp_path / "ok.jsonl", RECORDS))
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1, "title": "no abstract"}\n')
    with pytest.raises(DatasetError):
        validate_jsonl_sample(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(DatasetError):
        validate_jsonl_sample(empty)


def test_arxiv_url():
    assert arxiv_url("2509.21319v1") == "https://arxiv.org/abs/2509.21319v1"
    assert arxiv_url("hep-th/9901001") == "https://arxiv.org/abs/hep-th/9901001"
    assert arxiv_url("my-doc-7") is None
