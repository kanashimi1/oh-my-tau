| task | steps | strategy | model | rep | prompt uncached | prompt cached | cache write | completion | cache-hit % | latency s | calls | edits ok/rej | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1-mid | 6 | append | zai/glm-5.3-flash | 1 | 4495 | 27008 | 0 | 827 | 85.7 | 31.8 | 7 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | append | zai/glm-5.3-flash | 2 | 7545 | 23808 | 0 | 716 | 75.9 | 26.6 | 7 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | clm | zai/glm-5.3-flash | 1 | 2552 | 24640 | 0 | 536 | 90.6 | 60.6 | 12 | 5/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | clm | zai/glm-5.3-flash | 2 | 3221 | 25536 | 0 | 806 | 88.8 | 53.5 | 12 | 5/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | clm-host | zai/glm-5.3-flash | 1 | 2755 | 29760 | 0 | 730 | 91.5 | 78.8 | 13 | 5/1 | FAIL (anchors=0/8 wrong=0) |
| t1-mid | 30 | append | zai/glm-5.3-flash | 1 | 7742 | 184576 | 0 | 2729 | 96.0 | 113.4 | 31 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 30 | clm | zai/glm-5.3-flash | 1 | 20130 | 129536 | 0 | 2070 | 86.6 | 272.6 | 61 | 29/1 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 30 | clm-host | zai/glm-5.3-flash | 1 | 13299 | 122176 | 0 | 2059 | 90.2 | 219.7 | 60 | 29/0 | PASS (anchors=8/8 wrong=0) |
| t2-mid | 6 | append | zai/glm-5.3-flash | 1 | 5482 | 19072 | 0 | 32244 | 77.7 | 438.8 | 7 | 0/0 | FAIL (format miss) |
| t2-mid | 6 | append | zai/glm-5.3-flash | 2 | 5598 | 19456 | 0 | 32328 | 77.7 | 434.4 | 7 | 0/0 | FAIL (format miss) |
| t2-mid | 6 | clm | zai/glm-5.3-flash | 1 | 5374 | 21248 | 0 | 786 | 79.8 | 49.8 | 12 | 5/0 | FAIL (format miss) |
| t2-mid | 6 | clm | zai/glm-5.3-flash | 2 | 1960 | 23104 | 0 | 527 | 92.2 | 48.5 | 12 | 5/0 | FAIL (format miss) |
| t2-mid | 6 | clm-host | zai/glm-5.3-flash | 1 | 2182 | 25088 | 0 | 528 | 92.0 | 50.2 | 13 | 6/0 | FAIL (format miss) |
| t2-mid | 6 | clm-host | zai/glm-5.3-flash | 2 | 2160 | 23488 | 0 | 657 | 91.6 | 52.7 | 12 | 5/0 | FAIL (format miss) |
| t3-mid | 6 | append | zai/glm-5.3-flash | 1 | 3207 | 21696 | 0 | 502 | 87.1 | 24.8 | 7 | 0/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | append | zai/glm-5.3-flash | 2 | 5658 | 19776 | 0 | 600 | 77.8 | 24.3 | 7 | 0/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | clm | zai/glm-5.3-flash | 1 | 5013 | 21248 | 0 | 918 | 80.9 | 63.4 | 12 | 5/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | clm | zai/glm-5.3-flash | 2 | 3372 | 25792 | 0 | 981 | 88.4 | 54.2 | 13 | 6/0 | FAIL (pin_ok=False aborted=True) |
| t3-mid | 6 | clm-host | zai/glm-5.3-flash | 1 | 3799 | 32128 | 0 | 1041 | 89.4 | 62.6 | 15 | 6/2 | PASS (pin_ok=True aborted=False) |
| t3-mid | 6 | clm-host | zai/glm-5.3-flash | 2 | 3226 | 25472 | 0 | 954 | 88.8 | 60.4 | 13 | 6/0 | FAIL (pin_ok=True aborted=True) |

| task | steps | model | strategy | cache-hit % | Δ vs append (pp) | billed-ish tokens ratio | latency ratio | within 5 pp |
|---|---|---|---|---|---|---|---|---|
| t1-mid | 6 | zai/glm-5.3-flash | append | 80.8 | - | - | - | - |
| t1-mid | 6 | zai/glm-5.3-flash | clm | 89.7 | +8.9 | 0.89 | 1.95 | yes |
| t1-mid | 6 | zai/glm-5.3-flash | clm-host | 91.5 | +10.7 | 1.03 | 2.70 | yes |
| t1-mid | 30 | zai/glm-5.3-flash | append | 96.0 | - | - | - | - |
| t1-mid | 30 | zai/glm-5.3-flash | clm | 86.6 | -9.4 | 0.78 | 2.40 | NO |
| t1-mid | 30 | zai/glm-5.3-flash | clm-host | 90.2 | -5.8 | 0.71 | 1.94 | NO |
| t2-mid | 6 | zai/glm-5.3-flash | append | 77.7 | - | - | - | - |
| t2-mid | 6 | zai/glm-5.3-flash | clm | 86.0 | +8.3 | 0.46 | 0.11 | yes |
| t2-mid | 6 | zai/glm-5.3-flash | clm-host | 91.8 | +14.1 | 0.47 | 0.12 | yes |
| t3-mid | 6 | zai/glm-5.3-flash | append | 82.4 | - | - | - | - |
| t3-mid | 6 | zai/glm-5.3-flash | clm | 84.7 | +2.2 | 1.11 | 2.40 | yes |
| t3-mid | 6 | zai/glm-5.3-flash | clm-host | 89.1 | +6.7 | 1.30 | 2.51 | yes |
