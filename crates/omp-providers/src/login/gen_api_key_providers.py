#!/usr/bin/env python3
"""Regenerate api_key_providers.json from the TS catalog's `rules/auth/*.kdl`
(`login "api-key"` blocks) in `/login` roster order (`_order.kdl`).

usage: gen_api_key_providers.py <rules/auth dir> > api_key_providers.json
"""
import json, re, sys, pathlib

root = pathlib.Path(sys.argv[1])
Q = r'"((?:[^"\\]|\\.)*)"'

def unq(s):
    return json.loads('"' + s + '"')

def kv(line):
    return {m.group(1): (unq(m.group(2)) if m.group(2) is not None else m.group(3))
            for m in re.finditer(r'([a-z][a-z-]*)=(?:"((?:[^"\\]|\\.)*)"|(\S+))', line)}

order = re.search(r'login-order\s+(.*)', (root / '_order.kdl').read_text().replace('\\\n', ' '), re.S)
order = re.findall(Q, order.group(1))
order = [unq(o) for o in order]
out = {}
for f in sorted(root.glob('*.kdl')):
    if f.name.startswith('_'):
        continue
    text = f.read_text()
    m = re.search(r'^auth ' + Q, text, re.M)
    pid = unq(m.group(1))
    name = re.search(r'^\tname ' + Q, text, re.M)
    block = re.search(r'^\tlogin "api-key" \{\n(.*?)^\t\}', text, re.M | re.S)
    if not block:
        continue
    body = block.group(1)
    e = {'id': pid, 'name': unq(name.group(1))}
    for key, field in (('auth-url', 'auth_url'), ('instructions', 'instructions')):
        mm = re.search(r'^\t\t' + key + ' ' + Q, body, re.M)
        if mm: e[field] = unq(mm.group(1))
    mm = re.search(r'^\t\tprompt ' + Q + r'(.*)$', body, re.M)
    e['prompt'] = unq(mm.group(1))
    ph = kv(mm.group(2)).get('placeholder')
    if ph is not None: e['placeholder'] = ph
    mm = re.search(r'^\t\tnormalize ' + Q, body, re.M)
    if mm: e['normalize'] = unq(mm.group(1))
    mm = re.search(r'^\t\tempty-fallback ' + Q, body, re.M)
    if mm: e['empty_fallback'] = unq(mm.group(1))
    mm = re.search(r'^\t\tvalidate ' + Q + r'(.*)$', body, re.M)
    if mm:
        a = kv(mm.group(2))
        v = {'kind': unq(mm.group(1))}
        for k, f2 in (('label','label'),('base-url','base_url'),('url','url'),('model','model'),
                      ('max-tokens-field','max_tokens_field'),('base-url-env','base_url_env'),
                      ('headers-hook','headers_hook')):
            if k in a: v[f2] = a[k]
        if 'max-tokens' in a: v['max_tokens'] = int(a['max-tokens'])
        for k, f2 in (('optional','optional'),('trust-forbidden','trust_forbidden'),
                      ('tolerate-model-denied','tolerate_model_denied')):
            if a.get(k) == '#true': v[f2] = True
        e['validate'] = v
    out[pid] = e
rows = [out[p] for p in order if p in out] + [out[p] for p in sorted(out) if p not in order]
json.dump(rows, sys.stdout, indent=1, ensure_ascii=False)
print()
