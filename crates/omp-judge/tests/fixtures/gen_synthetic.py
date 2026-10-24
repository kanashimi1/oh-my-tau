#!/usr/bin/env python3
"""Generate golden_view_sel.jsonl: 50 synthetic episodes + expected judge views.

Usage:
  gen_synthetic.py --judge-view /path/to/judge_view.py [--out golden_view_sel.jsonl]
  (or set OMP_JUDGE_VIEW_PY instead of --judge-view)

`expected` is produced by the Python reference `judge_view.view_sel` (budget 2600); the Rust test
`tests/golden.rs` must reproduce it byte for byte. Every episode is fabricated from the word pools
below (seeded RNG, deterministic); no real session text is involved.

Record shape: {id, state, objective, expected}; state = "USER REQUEST:\n<req>\n\nSTEPS (action -> outcome):\n
1. tool: label | args -> [ok|ERR exit=N] output\n...\n\nFINAL REPLY:\n<reply>".
"""
import argparse
import importlib.util
import json
import os
import random
import sys

TOPICS_EN = [
    ("parser", "tokenizer", "lexer", "grammar", "ast"),
    ("cache", "eviction", "ttl", "shard", "lookup"),
    ("scheduler", "queue", "worker", "retry", "backoff"),
    ("config", "loader", "schema", "defaults", "validation"),
    ("renderer", "layout", "viewport", "scrollbar", "palette"),
    ("migration", "database", "index", "rollback", "transaction"),
    ("logger", "rotation", "formatter", "handler", "verbosity"),
    ("network", "socket", "timeout", "handshake", "proxy"),
]
TOPICS_RU = [
    ("парсер", "токенизатор", "грамматика", "разбор", "узел"),
    ("кэш", "вытеснение", "очередь", "поиск", "хранилище"),
    ("планировщик", "задача", "повтор", "таймаут", "воркер"),
    ("конфигурация", "схема", "значения", "проверка", "загрузчик"),
    ("интерфейс", "экран", "кнопка", "прокрутка", "палитра"),
]
REQ_EN = [
    "please fix the {a} so the {b} stops failing on empty input",
    "add a {b} to the {a} module and wire it into the {c}",
    "why does the {a} ignore the {b}? check the {c} too",
    "refactor the {a}, keep the {b} behaviour identical, update the {c} tests",
    "continue",
    "write a small {a} for the {b} and show how to run it",
]
REQ_RU = [
    "почему {a} игнорирует {b}? проверь ещё {c}",
    "исправь {a}, чтобы {b} не падал на пустом входе",
    "добавь {b} в модуль {a} и подключи к {c}",
    "продолжай",
    "перепиши {a} без изменения поведения {b}, обнови тесты на {c}",
]
TOOLS = ["bash", "read", "edit", "write", "eval", "grep", "find", "todo", "fetch"]
TEST_CMDS = ["cargo test -p demo", "pytest -q tests/", "npm run test", "go test ./...", "ruff check src", "mypy pkg", "tsc --noEmit"]
PLAIN_CMDS = ["ls -la {d}", "cat {d}/{f}", "git status --short", "rg -n {w} {d}", "sed -n 1,80p {d}/{f}", "wc -l {d}/{f}", "make -C {d} all"]
DIRS = ["src", "lib/core", "app/ui", "pkg/util", "docs", "scripts", "tests/unit"]
FILES = ["main.rs", "mod.py", "index.ts", "util.go", "config.toml", "README.md", "handler.rs"]
OK_EN = ["done", "42 lines", "no matches", "file updated", "3 passed, 0 failed", "ready", "total 12", "exit ok"]
OK_RU = ["готово", "строк: 42", "совпадений нет", "файл обновлён", "пройдено 3 теста", "всё хорошо"]
ERR_EN = ["No such file or directory", "permission denied", "assertion failed: left == right", "command not found", "timed out after 30s", "connection reset by peer"]
ERR_RU = ["файл не найден", "доступ запрещён", "ошибка сборки: неизвестный символ", "превышено время ожидания"]
FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore".split()
FILLER_RU = "съешь же ещё этих мягких французских булок да выпей чаю быстрая коричневая лиса прыгает через ленивую собаку".split()
EMOJI = ["🚀", "✅", "🔧", "𝒳", "日本語", "ñandú", "—", "«кавычки»"]
FINALS_EN = ["The {a} is fixed and the {b} now passes.", "Added the {b}; the {c} is wired up.", "I could not reproduce the failure, see notes above.",
             "Done. Summary:\n- changed the {a}\n- updated the {b}\n- left the {c} untouched"]
FINALS_RU = ["Исправил {a}, теперь {b} проходит.", "Добавил {b}, подключил {c}.", "Не удалось воспроизвести, подробности выше.",
             "Готово. Итог:\n- изменён {a}\n- обновлён {b}\n- {c} не трогал"]


def words(rng, pool, n):
    return " ".join(rng.choice(pool) for _ in range(n))


def make_episode(i, rng):
    ru = i % 3 == 1
    topic = rng.choice(TOPICS_RU if ru else TOPICS_EN)
    a, b, c = rng.sample(topic, 3)
    req = rng.choice(REQ_RU if ru else REQ_EN).format(a=a, b=b, c=c)
    if i % 7 == 0:  # request longer than the 600-char cap
        req += ". " + words(rng, FILLER_RU if ru else FILLER, 120)
    if i % 11 == 5:  # multi-line, quoted tool output pasted into the request
        req = "ошибка при запуске:\n  error: " + b + " failed\n  at " + a + ".rs:12\n" + req if ru else "build log:\n  error: " + b + " failed\n  at " + a + ".rs:12\n" + req
    # step count profile: empty, tiny, mid, budget-truncating, very many
    profile = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 120, 250]
    n = profile[i % len(profile)] + (i // len(profile)) * 3
    if i == 49:
        n = 400
    steps, errs, fail_run, fail_run_max, tests_run, tests_pass = [], 0, 0, 0, False, None
    last_err_msg = None
    for k in range(n):
        tool = rng.choice(TOOLS)
        word = rng.choice(topic)
        d, f = rng.choice(DIRS), rng.choice(FILES)
        if tool == "bash" and rng.random() < 0.3:
            cmd = rng.choice(TEST_CMDS)
            is_test = True
        elif tool == "bash":
            cmd = rng.choice(PLAIN_CMDS).format(d=d, f=f, w=word)
            is_test = False
        elif tool in ("edit", "write"):
            cmd = f"{d}/{f}"
            is_test = False
        elif tool == "eval":
            cmd = json.dumps({"language": "js", "code": f"console.log(await tool.read({{path:'{d}/{f}'}}))"})
            is_test = False
        else:
            cmd = f"{word} {d}"
            is_test = False
        if rng.random() < 0.1:
            cmd += " -> " + rng.choice(["out.log", "/dev/null"])  # arrow inside the action text
        if rng.random() < 0.15:
            cmd += " " + rng.choice(EMOJI)
        label = (words(rng, topic, 2) if not ru or rng.random() < 0.3 else words(rng, topic, 2) + " " + rng.choice(["чтение", "правка", "проверка"]))
        label = label.capitalize()
        fail = rng.random() < (0.45 if i % 5 == 2 else 0.12)
        if fail and fail_run and last_err_msg and rng.random() < 0.6:
            msg = last_err_msg  # identical consecutive failure
        elif fail:
            msg = rng.choice(ERR_RU if ru and rng.random() < 0.5 else ERR_EN)
        else:
            msg = rng.choice(OK_RU if ru and rng.random() < 0.5 else OK_EN)
        if rng.random() < 0.2:  # long output; steps are single-line (newlines flattened) like real states
            msg += " " + words(rng, FILLER_RU if ru else FILLER, rng.randint(40, 300))
        if fail:
            out = f"[ERR exit={rng.choice([1, 1, 2, 127])}] {msg}"
            errs += 1
            fail_run = fail_run + 1 if msg == last_err_msg else 1
            last_err_msg = msg
            fail_run_max = max(fail_run_max, fail_run)
        else:
            out = f"[ok] {msg}"
            fail_run, last_err_msg = 0, None
        if is_test:
            tests_run, tests_pass = True, not fail
        steps.append(f"{k + 1}. {tool}: {label} | {cmd} -> {out}")
    interrupted = n > 0 and i % 9 == 4
    reply_pool = FINALS_RU if ru else FINALS_EN
    if interrupted or i % 13 == 6:
        final = ""
    elif i % 8 == 3:  # final reply longer than the 1200-char cap
        final = rng.choice(reply_pool).format(a=a, b=b, c=c) + "\n" + words(rng, FILLER_RU if ru else FILLER, 260)
    else:
        final = rng.choice(reply_pool).format(a=a, b=b, c=c)
    state = "USER REQUEST:\n" + req + "\n\nSTEPS (action -> outcome):\n" + "\n".join(steps)
    if final:
        state += "\n\nFINAL REPLY:\n" + final
    obj = {
        "tests_run": tests_run,
        "tests_final_pass": tests_pass,
        "ended_with_error": bool(steps) and steps[-1].split(" -> ")[-1].startswith("[ERR"),
        "identical_fail_run_max": fail_run_max,
        "n_steps": n,
        "interrupted": interrupted,
        "claimed_done": bool(final) and i % 4 != 0,
        "is_subagent": i % 10 == 9,
        "model": "synthetic-model",
    }
    return {"id": f"synthetic-{i}", "state": state, "objective": obj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-view", default=os.environ.get("OMP_JUDGE_VIEW_PY"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_view_sel.jsonl"))
    ap.add_argument("--count", type=int, default=50)
    args = ap.parse_args()
    if not args.judge_view:
        sys.exit("pass --judge-view /path/to/judge_view.py (or set OMP_JUDGE_VIEW_PY)")
    spec = importlib.util.spec_from_file_location("judge_view", args.judge_view)
    jv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jv)
    rng = random.Random(20240601)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            r = make_episode(i, rng)
            r["expected"] = jv.view_sel(r, 2600)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
