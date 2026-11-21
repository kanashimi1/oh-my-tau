# SparseRead A/B: официальная реализация

## Вывод

На этом наборе синтетических coding-задач официальный SparseRead **не снизил суммарное потребление токенов LLM**. Все 48 ячеек дали правильный ответ, поэтому различие ниже относится к стоимости и задержке, а не к точности.

| Модель | Baseline tokens | SparseRead tokens | Δ tokens | Baseline time, s | SparseRead time, s | Δ time | Парных выигрышей по tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zai/glm-5.3-flash` | 216,987 | 305,821 | +40.9% | 107.9 | 214.1 | +98.4% | 3/12 |
| `kimi-code/kimi-for-coding` | 247,938 | 503,384 | +103.0% | 130.8 | 376.9 | +188.1% | 3/12 |

Токены — сумма provider-reported `input + cacheRead + cacheWrite + output` по всем ходам ячейки. Время — wall-clock одной ячейки. В таблице суммированы 4 задачи × 3 повтора.

## Среднее по задачам

| Модель | Задача | Baseline tokens | SparseRead tokens | Δ | Baseline time, s | SparseRead time, s |
|---|---|---:|---:|---:|---:|---:|
| `zai/glm-5.3-flash` | `config` | 35,227 | 14,306 | -59.4% | 12.0 | 9.4 |
| `zai/glm-5.3-flash` | `diagnose` | 12,633 | 19,693 | +55.9% | 6.3 | 15.0 |
| `zai/glm-5.3-flash` | `logs` | 6,419 | 19,764 | +207.9% | 8.5 | 16.6 |
| `zai/glm-5.3-flash` | `whole` | 18,049 | 48,177 | +166.9% | 9.2 | 30.3 |
| `kimi-code/kimi-for-coding` | `config` | 22,725 | 64,169 | +182.4% | 14.0 | 43.6 |
| `kimi-code/kimi-for-coding` | `diagnose` | 12,687 | 27,297 | +115.2% | 7.1 | 29.3 |
| `kimi-code/kimi-for-coding` | `logs` | 26,175 | 26,324 | +0.6% | 14.1 | 15.2 |
| `kimi-code/kimi-for-coding` | `whole` | 21,060 | 50,005 | +137.4% | 8.4 | 37.6 |

Единственное устойчивое снижение токенов во всех трёх повторах — GLM на `config`: в среднем **−59.4%**. Остальные парные результаты либо отрицательны, либо нестабильны между повторами.

## Что именно запускалось

- Официальный репозиторий: https://github.com/Liuzedooong/SparseReading
- Зафиксированный commit: `6749bb4a8b29d064e87638e18221e83f5aa10148`; сервер сообщил версию `0.1.4`.
- Официальный `SparseReadConfig(mode="auto")` через `packages/sparseread-core`; локальный адаптер `official_mcp.py` только переводит MCP stdio JSON-RPC в вызовы официальных tools.
- Инструменты: `sro_preview`, `sro_read`, `sro_raw`, `sro_card`. Самописный Rust `sparse_read` удалён из исходников и в прогонах не использовался.
- Sparse arm: `mcp__sro_preview,mcp__sro_read,mcp__sro_raw,mcp__sro_card,grep,glob`; baseline: `read,grep,glob`.
- Модели: `zai/glm-5.3-flash`, `kimi-code/kimi-for-coding`; `--thinking minimal`; по 3 независимых повтора.
- Задачи: извлечение production-конфигурации среди decoys (`config`), локализация off-by-one (`diagnose`), поиск последней fatal-ошибки (`logs`), полное чтение документа (`whole`).
- Детерминированный scorer, без сетевого judge. Каждая ячейка получала новый fixture и временный HOME; файлы агентами не изменялись.

## Проверка корректности

- 48 уникальных ячеек: 2 модели × 4 задачи × 2 arm × 3 повтора.
- `ok=true`, `score=1.0`, `tool_errors=0` во всех 48 ячейках.
- Sparse arm реально вызывал MCP tools; baseline их не вызывал. Обычный `read` в sparse arm отсутствовал.
- Во всех sparse-записях commit официального репозитория совпадает с указанным выше.

## Почему tool I/O и tokens расходятся

| Модель | Baseline tool output chars | SparseRead tool output chars | Δ | Baseline tool calls | SparseRead tool calls | `sro_raw` calls | Неограниченные `sro_raw` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zai/glm-5.3-flash` | 618,075 | 137,654 | -77.7% | 42 | 58 | 17 | 0 |
| `kimi-code/kimi-for-coding` | 567,630 | 332,746 | -41.4% | 35 | 102 | 45 | 7 |

`sro_raw` с `range` или `selector` — ограниченный raw-доступ, а не полный просмотр файла. Неограниченный raw засчитан только при отсутствии обоих параметров. Поле `full_view_fallbacks` в уже записанном run1/run2-3 JSONL исторически считало все raw-вызовы; таблица пересчитана из `tool_trace`, а harness для будущих запусков исправлен и дополнительно пишет `raw_calls`.

**Интерпретация (не отдельный эксперимент):** SparseRead действительно уменьшил текст, возвращаемый инструментами, но дополнительные ходы, схемы/результаты MCP и повторные уточнения перевесили экономию. У Kimi это особенно заметно на `config`; у GLM — на `whole`. Это согласуется с ростом tool calls и общего LLM token usage при снижении tool output chars.

## Ограничения

- Только синтетические текстовые fixtures умеренного размера; это не воспроизведение всех сценариев и корпусов статьи.
- Три повтора; высокая межповторная дисперсия не позволяет делать статистически сильный вывод по отдельным задачам.
- Протестирован `auto`, а не специально настроенный `bench_protocol`; протокол и параметры официального reader не тюнились под модели.
- Нельзя переносить проценты на большие репозитории, PDF или другие модели без отдельного прогона.

## Данные и воспроизведение

- Сырые записи: `bench/sparseread/results/run1.jsonl` (repeat 1) и `bench/sparseread/results/run2-3.jsonl` (repeats 2–3). Каждая запись содержит usage, ответ, score и полный `tool_trace`.
- Harness: `bench/sparseread/run.py`; MCP adapter: `bench/sparseread/official_mcp.py`.

```bash
git clone https://github.com/Liuzedooong/SparseReading /tmp/SparseReading-official
git -C /tmp/SparseReading-official checkout 6749bb4a8b29d064e87638e18221e83f5aa10148
python3 bench/sparseread/run.py --start-repeat 1 --repeat 3 --timeout 300 --output bench/sparseread/results/repro.jsonl
```
