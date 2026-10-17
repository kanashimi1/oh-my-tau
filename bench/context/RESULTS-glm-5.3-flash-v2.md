| task | steps | strategy | model | rep | prompt uncached | prompt cached | cache write | completion | cache-hit % | latency s | calls | edits ok/rej | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1-mid | 6 | append | zai/glm-5.3-flash | 1 | 4495 | 27008 | 0 | 827 | 85.7 | 31.8 | 7 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | append | zai/glm-5.3-flash | 2 | 7545 | 23808 | 0 | 716 | 75.9 | 26.6 | 7 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | clm | zai/glm-5.3-flash | 1 | 12999 | 19712 | 0 | 760 | 60.3 | 49.3 | 12 | 5/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | clm | zai/glm-5.3-flash | 2 | 3803 | 28288 | 0 | 893 | 88.1 | 48.5 | 12 | 5/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 6 | clm-host | zai/glm-5.3-flash | 1 | 3775 | 28416 | 0 | 732 | 88.3 | 45.7 | 12 | 5/0 | FAIL (anchors=0/8 wrong=0) |
| t1-mid | 6 | clm-host | zai/glm-5.3-flash | 2 | 7949 | 27968 | 0 | 886 | 77.9 | 46.9 | 13 | 5/1 | FAIL (anchors=0/8 wrong=0) |
| t1-mid | 30 | append | zai/glm-5.3-flash | 1 | 10014 | 163136 | 0 | 1457 | 94.2 | 94.5 | 31 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 30 | append | zai/glm-5.3-flash | 1 | 7742 | 184576 | 0 | 2729 | 96.0 | 113.4 | 31 | 0/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 30 | clm | zai/glm-5.3-flash | 1 | 21231 | 120832 | 0 | 2720 | 85.1 | 198.4 | 60 | 29/0 | PASS (anchors=8/8 wrong=0) |
| t1-mid | 30 | clm-host | zai/glm-5.3-flash | 1 | 14154 | 130944 | 0 | 2122 | 90.2 | 167.5 | 60 | 29/0 | PASS (anchors=8/8 wrong=0) |
| t2-mid | 6 | append | zai/glm-5.3-flash | 1 | 5482 | 19072 | 0 | 32244 | 77.7 | 438.8 | 7 | 0/0 | FAIL (format miss) |
| t2-mid | 6 | append | zai/glm-5.3-flash | 2 | 5598 | 19456 | 0 | 32328 | 77.7 | 434.4 | 7 | 0/0 | FAIL (format miss) |
| t3-mid | 6 | append | zai/glm-5.3-flash | 1 | 3207 | 21696 | 0 | 502 | 87.1 | 24.8 | 7 | 0/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | append | zai/glm-5.3-flash | 2 | 5658 | 19776 | 0 | 600 | 77.8 | 24.3 | 7 | 0/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | clm | zai/glm-5.3-flash | 1 | 5762 | 25600 | 0 | 780 | 81.6 | 39.5 | 13 | 6/0 | PASS (pin_ok=True aborted=False) |
| t3-mid | 6 | clm | zai/glm-5.3-flash | 2 | 4153 | 25984 | 0 | 940 | 86.2 | 47.8 | 12 | 5/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | clm-host | zai/glm-5.3-flash | 1 | 7349 | 34432 | 0 | 1604 | 82.4 | 58.3 | 13 | 6/0 | FAIL (pin_ok=True aborted=True) |
| t3-mid | 6 | clm-host | zai/glm-5.3-flash | 2 | 6516 | 34752 | 0 | 2267 | 84.2 | 68.1 | 12 | 4/1 | FAIL (pin_ok=True aborted=True) |

| task | steps | model | strategy | cache-hit % | Δ vs append (pp) | billed-ish tokens ratio | latency ratio | within 5 pp |
|---|---|---|---|---|---|---|---|---|
| t1-mid | 6 | zai/glm-5.3-flash | append | 80.8 | - | - | - | - |
| t1-mid | 6 | zai/glm-5.3-flash | clm | 74.2 | -6.6 | 1.03 | 1.67 | NO |
| t1-mid | 6 | zai/glm-5.3-flash | clm-host | 83.1 | +2.3 | 1.08 | 1.59 | yes |
| t1-mid | 30 | zai/glm-5.3-flash | append | 95.1 | - | - | - | - |
| t1-mid | 30 | zai/glm-5.3-flash | clm | 85.1 | -10.0 | 0.78 | 1.91 | NO |
| t1-mid | 30 | zai/glm-5.3-flash | clm-host | 90.2 | -4.9 | 0.80 | 1.61 | yes |
| t2-mid | 6 | zai/glm-5.3-flash | append | 77.7 | - | - | - | - |
| t3-mid | 6 | zai/glm-5.3-flash | append | 82.4 | - | - | - | - |
| t3-mid | 6 | zai/glm-5.3-flash | clm | 83.9 | +1.5 | 1.23 | 1.78 | yes |
| t3-mid | 6 | zai/glm-5.3-flash | clm-host | 83.3 | +0.9 | 1.69 | 2.57 | yes |
