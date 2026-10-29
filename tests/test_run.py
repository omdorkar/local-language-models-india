"""Harness tests with a mocked OpenAI-compatible server (httpx.MockTransport): no real model is called."""

import json

import httpx
import pytest

from evaluation import client as client_mod
from evaluation import run
from evaluation.client import parse_answer
from evaluation.config import MODELS
from evaluation.prompts import GMMLU_TEMPLATE_SHA, TEMPLATE_SHA, belebele_gold

SETTINGS = run.MODEL_SETTINGS["llama3.1-8b"]


def row(n: int) -> dict:
    return {"flores_passage": f"passage {n}", "question": f"q {n}", "mc_answer1": "a", "mc_answer2": "b",
            "mc_answer3": "c", "mc_answer4": "d", "correct_answer_num": "2"}


ITEMS = [f"link{n}#1" for n in range(6)]
DATA = {"eng_Latn": {i: row(n) for n, i in enumerate(ITEMS)},
        "hin_Deva": {i: row(n) for n, i in enumerate(ITEMS)}}


class Server:
    """Mock chat server: answers with `reply` (a callable of the request body) and records requests."""

    def __init__(self, reply=lambda body: '{"answer": "B"}'):
        self.requests, self.reply = [], reply
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": self.reply(body)}}],
                                         "usage": {"prompt_tokens": 7, "completion_tokens": 3}})


def go(tmp_path, server, conditions=("eng_Latn", "hin_Deva"), items=ITEMS, settings=SETTINGS, data=DATA):
    return run.run_model("belebele", settings, data, list(conditions), list(items), server.client,
                         runs_dir=tmp_path, sleep=lambda s: None, progress=False)


def rows_of(tmp_path, key="llama3.1-8b"):
    return run.read_rows(run.run_path("belebele", key, tmp_path))


# ---- parsing

@pytest.mark.parametrize("raw,expected", [
    ('{"answer": "A"}', "A"), ('{"answer": "D"}', "D"), ('{"answer": "E"}', None), ('{"answer": "a"}', None),
    ('{"answer": "AB"}', None), ('{"answer": ""}', None), ('{"answer": ["A"]}', None), ('{"x": "A"}', None),
    ("A", None), ("", None), ("[1]", None), ("null", None),
])
def test_parse_answer(raw, expected):
    assert parse_answer(raw) == expected


def test_invalid_answer_counts_as_wrong(tmp_path):
    go(tmp_path, Server(reply=lambda b: "I think it is B"), items=ITEMS[:2], conditions=["eng_Latn"])
    rows = [r for r in rows_of(tmp_path) if r["condition"] != "__template__"]
    assert len(rows) == 2 and all(r["answer"] is None and r["correct"] is False and r["gold"] == "B" for r in rows)


def test_request_settings_and_row_fields(tmp_path):
    for key, extra in [("qwen3-8b", ("none", 32)), ("gpt-oss-20b", ("low", 512)), ("llama3.1-8b", (None, 32))]:
        s = run.MODEL_SETTINGS[key]
        srv = Server()
        go(tmp_path, srv, items=ITEMS[:1], conditions=["hin_Deva"], settings=s)
        body = srv.requests[-1]
        assert body["temperature"] == 0 and body["seed"] == 1909 and body["model"] == s.tag
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["response_format"]["type"] == "json_schema"
        assert (body.get("reasoning_effort"), body["max_tokens"]) == extra
    r = [r for r in rows_of(tmp_path) if r["condition"] == "hin_Deva"][0]
    assert r | {"raw": 0, "seconds": 0, "timestamp": 0} == {
        "benchmark": "belebele", "model_key": "llama3.1-8b", "model_tag": "llama3.1:8b", "condition": "hin_Deva",
        "lang": "hi", "script": "Deva", "source": "human", "item_id": "link0#1", "gold": "B", "answer": "B",
        "correct": True, "raw": 0, "prompt_tokens": 7, "completion_tokens": 3, "seconds": 0,
        "template_sha": TEMPLATE_SHA, "seed": 1909, "timestamp": 0}


# ---- resumability, calibration, errors

def test_resume_does_nothing_second_time(tmp_path):
    srv = Server()
    s1 = go(tmp_path, srv)
    n_first = len(srv.requests)
    assert s1["written"] == 12 and n_first == 13  # 12 rows + calibration
    s2 = go(tmp_path, srv)
    assert s2 == {"written": 0, "failed": 0, "calibration": 0} and len(srv.requests) == n_first
    assert len(rows_of(tmp_path)) == 13


def test_resume_fills_only_missing_rows(tmp_path):
    srv = Server()
    go(tmp_path, srv)
    path = run.run_path("belebele", "llama3.1-8b", tmp_path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-3]) + "\n" + lines[-3][:20])  # drop 3 rows, leave a truncated line
    before = len(srv.requests)
    s = go(tmp_path, srv)
    assert s["written"] == 3 and len(srv.requests) == before + 3
    keys = [(r["condition"], r["item_id"]) for r in rows_of(tmp_path)]
    assert len(keys) == len(set(keys)) == 13


def test_calibration_row_written_once(tmp_path):
    srv = Server()
    go(tmp_path, srv, items=ITEMS[:1], conditions=["eng_Latn"])
    go(tmp_path, srv, items=ITEMS[:3], conditions=["eng_Latn"])
    cal = [r for r in rows_of(tmp_path) if r["condition"] == "__template__"]
    assert len(cal) == 1
    c = cal[0]
    assert (c["item_id"], c["gold"], c["correct"], c["lang"], c["script"], c["source"]) == \
           ("__template__", "", False, "", "", "")
    first_prompt = srv.requests[0]["messages"][0]["content"]
    assert first_prompt == run.calibration_prompt("belebele") and "Passage:\n\n" in first_prompt


def test_failed_rows_logged_not_written_and_retried(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        body = json.loads(request.content)
        if "q 0" in body["messages"][0]["content"]:
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "B"}'}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleeps = []
    stats = run.run_model("belebele", SETTINGS, DATA, ["eng_Latn"], ITEMS[:2], client, runs_dir=tmp_path,
                          sleep=sleeps.append, progress=False)
    assert stats["failed"] == 1 and stats["written"] == 1
    assert sleeps == [2, 4, 8]  # 3 retries with backoff
    assert "link0#1" in (tmp_path / "errors.log").read_text()
    assert ("eng_Latn", "link0#1") not in run.done_keys(run.run_path("belebele", "llama3.1-8b", tmp_path))
    # a rerun with a healthy server fills the gap
    stats = go(tmp_path, Server(), items=ITEMS[:2], conditions=["eng_Latn"])
    assert stats["written"] == 1
    assert [r["prompt_tokens"] for r in rows_of(tmp_path) if r["item_id"] == "link0#1"] == [7]


# ---- determinism

def test_shuffle_deterministic_and_complete():
    pairs = run.all_pairs(["a", "b", "c"], [f"i{n}" for n in range(20)])
    s1, s2 = run.shuffled(pairs), run.shuffled(pairs)
    assert s1 == s2 != pairs and sorted(s1) == sorted(pairs)
    assert run.shuffled(pairs, seed=1) != s1


def test_request_order_is_the_shuffled_order(tmp_path):
    srv = Server()
    go(tmp_path, srv)
    order = [r["item_id"] for r in rows_of(tmp_path) if r["condition"] != "__template__"]
    expected = [i for c, i in run.shuffled(run.all_pairs(["eng_Latn", "hin_Deva"], ITEMS))]
    assert order == expected


# ---- conditions

def test_human_conditions():
    assert len(run.human_conditions("belebele")) == 18 and "eng_Latn" in run.human_conditions("belebele")
    assert run.human_conditions("gmmlu_lite") == ["en", "hi", "bn", "or"]
    assert sorted(run.human_conditions("gmmlu")) == sorted(["en", "hi", "bn", "te", "ne"])


@pytest.mark.parametrize("bench,cond,expected", [
    ("belebele", "eng_Latn", ("en", "Latn", "human")), ("belebele", "hin_Deva", ("hi", "Deva", "human")),
    ("belebele", "urd_Latn", ("ur", "Latn", "human")), ("belebele", "urd_Arab", ("ur", "Arab", "human")),
    ("belebele", "hin_Latn+xlit", ("hi", "Deva", "machine")), ("belebele", "urd_Latn+xlit", ("ur", "Arab", "machine")),
    ("belebele", "npi_Latn+xlit", ("ne", "Deva", "machine")),
    ("belebele", "tel_Telu+mt_en", ("en", "Latn", "machine")), ("belebele", "hin_Latn+cm", ("hi", "Latn", "machine")),
    ("gmmlu", "te", ("te", "Telu", "human")), ("gmmlu_lite", "or", ("or", "Orya", "human")),
    ("gmmlu", "ne", ("ne", "Deva", "human")), ("gmmlu", "hi+mt_en", ("en", "Latn", "machine")),
])
def test_condition_meta(bench, cond, expected):
    assert tuple(run.condition_meta(bench, cond).values()) == expected


def test_derived_condition_loading(tmp_path):
    d = tmp_path / "belebele"
    d.mkdir()
    rows = [{**row(n), "item_id": ITEMS[n], "source_condition": "hin_Deva", "method": "mt", "method_version": "1"}
            for n in (0, 1, 2)]
    (d / "hin_Deva+mt_en.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    loaded = run.load_condition("belebele", "hin_Deva+mt_en", ITEMS[:2] + ["not-sampled#1"], derived_dir=tmp_path)
    assert sorted(loaded) == ITEMS[:2] and loaded[ITEMS[0]]["question"] == "q 0"
    with pytest.raises(FileNotFoundError):
        run.load_condition("belebele", "hin_Latn+xlit", ITEMS, derived_dir=tmp_path)
    # the harness treats it like a human condition, but labels it machine / English
    srv = Server()
    run.run_model("belebele", SETTINGS, {"hin_Deva+mt_en": loaded}, ["hin_Deva+mt_en"], ITEMS[:3], srv.client,
                  runs_dir=tmp_path / "out", sleep=lambda s: None, progress=False)
    rs = [r for r in run.read_rows(run.run_path("belebele", "llama3.1-8b", tmp_path / "out")) if r["item_id"] != "__template__"]
    assert len(rs) == 2  # item 2 absent from `loaded`
    assert {(r["condition"], r["lang"], r["script"], r["source"]) for r in rs} == {("hin_Deva+mt_en", "en", "Latn", "machine")}


def test_real_belebele_item_ids_and_gold():
    if not (run.SAMPLES / "belebele.json").exists():
        pytest.skip("samples not drawn")
    ids = run.sampled_items("belebele", 2)
    eng = run.load_condition("belebele", "eng_Latn", ids)
    hin = run.load_condition("belebele", "hin_Latn", ids)
    assert sorted(eng) == sorted(hin) == sorted(ids)
    assert all(belebele_gold(eng[i]) == belebele_gold(hin[i]) for i in ids)


def test_gmmlu_prompt_and_rows(tmp_path):
    g = {"en": {"astronomy/test/1": {"question": "Q?", "option_a": "w", "option_b": "x", "option_c": "y",
                                      "option_d": "z", "answer": "C"}}}
    srv = Server(reply=lambda b: '{"answer": "C"}')
    run.run_model("gmmlu", SETTINGS, g, ["en"], ["astronomy/test/1"], srv.client, runs_dir=tmp_path,
                  sleep=lambda s: None, progress=False)
    prompt = srv.requests[-1]["messages"][0]["content"]
    assert "Passage" not in prompt and prompt.endswith("A. w\nB. x\nC. y\nD. z") and "Question:\nQ?" in prompt
    r = run.read_rows(run.run_path("gmmlu", "llama3.1-8b", tmp_path))[-1]
    assert (r["gold"], r["correct"], r["template_sha"], r["lang"], r["script"]) == ("C", True, GMMLU_TEMPLATE_SHA, "en", "Latn")
    assert GMMLU_TEMPLATE_SHA != TEMPLATE_SHA and TEMPLATE_SHA == "d3ee3d7bd6e7"


# ---- settings

def test_settings_cover_config_models():
    assert set(run.MAIN_MODELS) <= set(run.MODEL_SETTINGS) and "llama3.1-8b-q3" in run.MODEL_SETTINGS
    for m in MODELS:
        if m.ollama_tag:
            assert run.MODEL_SETTINGS[m.key].tag == m.ollama_tag
    assert run.MODEL_SETTINGS["sarvam-30b"].runtime == "llama-server"
    assert run.MODEL_SETTINGS["llama3.1-8b-q3"].tag == "llama3.1:8b-instruct-q3_K_M"


def test_smoke_uses_shared_ask():
    from evaluation import smoke
    assert smoke.ask is client_mod.ask
