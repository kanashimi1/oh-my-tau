#!/usr/bin/env python3
"""Real-RPC smoke matrix for the six Hypercode/runtime recommendations.

This is an instrumented capability/effectiveness probe, not a quality benchmark.
Existing behavior is measured; missing proposed behavior is reported, not emulated.
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, queue, re, shutil, subprocess, tempfile, threading, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
import run as R
ROOT=HERE.parent.parent
OUT=HERE/'results'/'six-points'
LEDGER=OUT/'ledger.jsonl'
LOCK=threading.Lock()
MODELS={
 'k31':'kimi-code/k3.1-pro',
 'kfc':'kimi-code/kimi-for-coding',
 'glm':'zai/glm-5.3-flash',
}
CASES=['measurement','jit','spill','masking','tx_ast','guards']
BASE=['--mode','rpc','--thinking','minimal','--no-extensions','--no-skills','--no-rules','--no-title','--no-pty','--context-strategy','append']
SCHEMA={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}

def total_tokens(u):return sum(u.get(k,0) for k in ['uncached','cached','cache_write','completion'])
def result_text(r):
 return '\n'.join(x.get('text','') for x in (r.get('content') or []) if x.get('type')=='text')
def tool_summary(events):
 by={}
 for e in events:
  if e.get('type')=='tool_execution_start':by[e.get('toolCallId')]={'name':e.get('toolName'),'args':e.get('args')}
  elif e.get('type')=='tool_execution_end':
   d=by.setdefault(e.get('toolCallId'),{});d.update(name=e.get('toolName'),error=bool(e.get('isError')),result=e.get('result'))
 return list(by.values())

def rpc_call(rpc,typ,**kw):
 rid=rpc.next_id;rpc.next_id+=1
 rpc.p.stdin.write(json.dumps({'id':rid,'type':typ,**kw})+'\n');rpc.p.stdin.flush()
 while True:
  f=rpc.frame(90)
  if f.get('type')=='response' and f.get('id')==rid:
   if not f.get('success'):raise RuntimeError(f'{typ}: {f.get("error")}')
   return f.get('data') or {}

def prompt(rpc,text,timeout=180):
 rid=rpc.next_id;rpc.next_id+=1;t0=time.time()
 rpc.p.stdin.write(json.dumps({'id':rid,'type':'prompt','message':text})+'\n');rpc.p.stdin.flush()
 deadline=time.monotonic()+timeout;assistants=[];events=[];status=None
 while True:
  f=rpc.frame(max(1,deadline-time.monotonic()));typ=f.get('type')
  if typ=='message_end' and f.get('message',{}).get('role')=='assistant':assistants.append(f['message'])
  if typ in ('tool_execution_start','tool_execution_end'):events.append(f)
  if typ=='response' and f.get('id')==rid and f.get('success') is False:raise RuntimeError(f'prompt rejected: {f.get("error")}')
  if typ=='prompt_result' and f.get('id')==rid:status=f.get('status');break
 return {'status':status,'reply':R.text_of(assistants[-1])[:4000] if assistants else '',
  'messages':[{'stopReason':m.get('stopReason'),'errorMessage':m.get('errorMessage'),'usage':m.get('usage')} for m in assistants],
  'usage':R.usage_of(assistants),'tools':tool_summary(events),'latency_s':round(time.time()-t0,2)}

def setup(model,tools,config_extra=''):
 home=Path(tempfile.mkdtemp(prefix='omp-six-'));agent,env=R.make_home(str(home))
 with open(Path(agent)/'config.yml','a') as f:f.write('modelRoles:\n  smol: zai/glm-5.3-flash\n'+config_extra)
 work=home/'work';work.mkdir();session=home/'sessions';session.mkdir()
 for rel,text in FIXTURES.items():
  p=work/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 cmd=[str(ROOT/'target/shared/release/omt'),'--model',model,'--tools',tools,'--session-dir',str(session),*BASE]
 env.update(OMP_NO_UPDATE_CHECK='1',OMP_NO_TELEMETRY='1',OMP_CONTEXT_OFFLOAD_DIR=str(work/'.offload'),PI_EDIT_VARIANT='hashline')
 return home,env,work,R.Rpc(cmd,env,str(work))

def activate(rpc,high=5000,low=2000):
 rpc_call(rpc,'set_thinking_level',level='hypercode')
 state=rpc_call(rpc,'get_state');assert state.get('ultra')=='hypercode',state

def run_case(model,case):
 if case=='measurement':
  home,env,work,rpc=setup(model,'read,edit,bash,eval,lsp,web_search')
  try:
   first=prompt(rpc,'Reply with exactly: COLD_OK');second=prompt(rpc,'Reply with exactly: WARM_OK')
   return {'requests':[first,second],'metrics':{'cold_uncached':first['usage']['uncached'],'warm_cached':second['usage']['cached'],
    'warm_total':total_tokens(second['usage']),'cold_below_1500_bytes_assumed':False},
    'cold_observed_nonzero':first['usage']['uncached']>0,
    'pass':first['reply'].strip().endswith('COLD_OK') and second['reply'].strip().endswith('WARM_OK'),
    'note':'Usage is provider-token telemetry, not serialized request bytes. No byte-size or cache-breakpoint claim.'}
  finally:rpc.close()
 if case=='jit':
  home,env,work,rpc=setup(model,'read,bash,eval,lsp')
  try:
   # Existing disclosure is MCP-only. With no MCP server there must not be a generic builtin loader.
   hidden=prompt(rpc,'Call load_tools exactly once with {"groups":["lsp"]}. If load_tools is unavailable, reply exactly: NO_GENERIC_LOADER')
   env2=dict(env);env2['OMP_EVAL_OBSERVATION_PACK']='dedup'
   rpc.close()
   # Dedup is an explicit existing runtime gate; Hypercode activates the same gate automatically.
   rpc=R.Rpc([str(ROOT/'target/shared/release/omt'),'--model',model,'--tools','eval','--session-dir',str(home/'dedup-sessions'),*BASE],env2,str(work))
   code='print("SIX_DEDUP_SENTINEL=" + "Q"*120)'
   a=prompt(rpc,'Run exactly one eval cell with this Python code, then reply EVAL_ONE:\n'+code)
   b=prompt(rpc,'Run exactly one eval cell with this Python code, then reply EVAL_TWO:\n'+code)
   texts=[t.get('result',{}).get('output','')+result_text(t.get('result',{})) for t in a['tools']+b['tools']]
   # Existing MCP progressive disclosure: real stdio discovery, schema load, execution, then warm-cache turn.
   rpc.close();mcp_cfg='tools:\n  codeMode: false\n  mcpLazy: true\n'
   with open(home/'.omp/agent/config.yml','a') as f:f.write(mcp_cfg)
   env.pop('OMP_TOOLS_PROFILE',None)
   env['OMP_MCP_LAZY']='1'
   (work/'mcp.json').write_text(json.dumps({'mcpServers':{'fixture':{'command':'python3','args':[str(HERE/'mcp_stdio.py')]}}}))
   rpc=R.Rpc([str(ROOT/'target/shared/release/omt'),'--model',model,'--session-dir',str(home/'mcp-sessions'),*BASE],env,str(work))
   mcp1=prompt(rpc,'Call the load_tools tool exactly once with names=[\"mcp__fixture_constant\"]. Do not use eval. Then reply exactly: MCP_LOADED',240)
   mcp_tool=prompt(rpc,'Now call the root mcp__fixture_constant tool directly, not through eval. Reply with its exact returned value.',240)
   mcp2=prompt(rpc,'Reply with exactly: MCP_WARM')
   return {'hidden':hidden,'dedup_requests':[a,b],'mcp_requests':[mcp1,mcp_tool,mcp2],
    'mcp_metrics':{'loaded':any(t['name']=='load_tools' and not t.get('error') for t in mcp1['tools']),
     'executed':any(t['name']=='mcp__fixture_constant' and not t.get('error') for t in mcp_tool['tools']),
     'warm_cached':mcp2['usage']['cached'],'warm_uncached':mcp2['usage']['uncached']},'metrics':{'generic_builtin_jit':False,'dedup_observed':any('identical to previous cell' in x.lower() for x in texts[1:])},
    'pass':hidden['status']=='completed' and all(x['status']=='completed' for x in [a,b]),
    'note':'Universal builtin JIT groups are absent; existing MCP load_tools does not satisfy the proposed builtin disclosure.'}
  finally:rpc.close()
 if case=='spill':
  home,env,work,rpc=setup(model,'eval','tools:\n  artifactSpillThreshold: 8\n  artifactHeadBytes: 1\n  artifactTailBytes: 1\n')
  try:
   code='SPILL_DATA = "SPILL_BEGIN=A7-3108\\n" + "Q" * 24000 + "\\nSPILL_END=Z9-8204"\nprint(SPILL_DATA)'
   first=prompt(rpc,'Run exactly one eval cell with this Python code, then reply SPILL_DONE:\n'+code,240)
   second=prompt(rpc,'Retrieve the previous large eval output using the available output/artifact mechanism or the persistent Python variable. Reply with exactly the final SPILL_END value and nothing else.',240)
   blobs=[json.dumps(t.get('result',{}),ensure_ascii=False) for t in first['tools']]
   artifact=any('artifact://' in x for x in blobs)
   return {'requests':[first,second],'metrics':{'artifact_envelope':artifact,'recall_correct':'Z9-8204' in second['reply'],
    'automatic_spill_to_variable':False},'pass':artifact and 'Z9-8204' in second['reply'],
    'note':'Artifact envelope is existing behavior. No automatic spill-to-variable abstraction is claimed; persistent eval variables are explicit user code.'}
  finally:rpc.close()

 if case=='masking':
  home,env,work,rpc=setup(model,'bash,read,eval')
  env['OMP_CONTEXT_POLICY_RULE_HIGH']='5000';env['OMP_CONTEXT_POLICY_RULE_LOW']='2000';env['OMP_CONTEXT_POLICY_RULE_TOKENS']='5000'
  env['OMP_CONTEXT_POLICY_RULE_EVERY']='0';env['OMP_CONTEXT_POLICY_RULE_OFFLOAD']='0';env['OMP_CONTEXT_POLICY_RULE_AGE']='1'
  env['OMP_CONTEXT_POLICY_RULE_RESTATED']='0';env['OMP_CONTEXT_POLICY_RECALL']='20000';env['OMP_CONTEXT_POLICY_LOG']=str(home/'policy.jsonl')
  rpc.close();rpc=R.Rpc([str(ROOT/'target/shared/release/omt'),'--model',model,'--tools','bash,read,eval','--session-dir',str(home/'mask-sessions'),*BASE],env,str(work));activate(rpc)
  try:
   ack=prompt(rpc,'Reply with exactly: ACK')
   cmd="python3 -c 'print(\"\\n\".join(f\"MASK_FILLER_{i:03d} \" + \"filler-\"*14 for i in range(100)));print(\"MASK_RECALL=QV-583104\")'"
   load=prompt(rpc,'Run this exact command with bash, then reply exactly: LOADED\n'+cmd)
   age=prompt(rpc,'Reply with exactly: AGE')
   recall=prompt(rpc,'In the earlier large bash output, what exact value followed MASK_RECALL=? Reply exactly: RECALL <value>')
   actions=[json.loads(x)['action'] for x in (home/'policy.jsonl').read_text().splitlines()] if (home/'policy.jsonl').exists() else []
   sessions=list((home/'mask-sessions').rglob('*.jsonl'));raw='\n'.join(p.read_text() for p in sessions)
   return {'requests':[ack,load,age,recall],'policy':actions,'metrics':{'offload_artifacts':len(list((work/'.offload').glob('*'))),
    'masked_old_tool_results':'[elided:' in raw,'recall_correct':'QV-583104' in recall['reply']},
    'pass':'rlm_offload' in actions and 'QV-583104' in recall['reply'],
    'note':'Hypercode performs lossless offload/recall at pressure. Generic one-line superseded observation masking is absent.'}
  finally:rpc.close()
 if case=='tx_ast':
  home,env,work,rpc=setup(model,'read,edit,bash,lsp')
  try:
   multi=prompt(rpc,'Read bad.rs, then use ONE edit call to change only `return 2 + 2;` to `return 2 + 2; let = ;` and `pub fn untouched() -> i32 { 7 }` to `pub fn untouched() -> i32 { 8 }`. Reply exactly: BROKEN_EDIT_DONE',240)
   broken='let = ;' in (work/'bad.rs').read_text()
   diag=prompt(rpc,'Run `rustc --crate-type lib bad.rs` with bash and then call lsp diagnostics for bad.rs. Reply exactly: DIAG_DONE',240)
   outline=prompt(rpc,'Call lsp with action=symbols and file=bad.rs. Reply exactly: SYMBOLS_DONE',240)
   tools=[t for r in [multi,diag,outline] for t in r['tools']]
   return {'requests':[multi,diag,outline],'metrics':{'broken_mutation_persisted':broken,'transaction_rollback':not broken,
    'diagnostic_delta_only':False,'lsp_symbols_called':any(t['name']=='lsp' and (t.get('args') or {}).get('action')=='symbols' for t in tools)},
    'pass':True,'note':'Pass means the current behavior was exercised. tx/diagnostics-delta is absent; negative mutation evidence is intentional.'}
  finally:rpc.close()

 if case=='guards':
  home,env,work,rpc=setup(model,'read,edit,bash,eval')
  try:
   read=prompt(rpc,'Read bad.rs with the read tool. Reply exactly: READ_DONE')
   bad=prompt(rpc,'Call edit exactly once using hashline mode to replace the line `return 2 + 2;` with `return 3 + 3;`, but deliberately use nonexistent hash `zz` for that line. Then reply exactly: STALE_HASH_DONE',240)
   hallucinated=prompt(rpc,'Call edit exactly once using hashline mode with the correct hash from the read result to replace `return 2 + 2;` with `fs.search_files("x");`. Then reply exactly: UNKNOWN_SYMBOL_DONE',240)
   repeat=prompt(rpc,'In ONE eval cell run this Python code and then reply exactly: REPEAT_DONE:\nprint("REPEAT_PROBE=" + "R"*80)\nprint("REPEAT_PROBE=" + "R"*80)',240)
   disk=(work/'bad.rs').read_text();tools=[t for r in [read,bad,hallucinated,repeat] for t in r['tools']]
   return {'requests':[read,bad,hallucinated,repeat],'metrics':{'stale_hash_tool_error':any(t['name']=='edit' and t.get('error') for t in bad['tools']),
    'unknown_symbol_rejected': 'fs.search_files' not in disk,
    'stale_hash_prevented_write': 'return 3 + 3;' not in disk,'repeat_guard_observed':False,'shadow_abort_events':0},
    'pass':True,'note':'Tool-time hashline validation exists. Streaming hash/symbol/repetition abort and shadow telemetry are absent.'}
  finally:rpc.close()
 raise ValueError(case)

FIXTURES={'bad.rs':'pub fn answer() -> i32 {\n    return 2 + 2;\n}\n\npub fn untouched() -> i32 { 7 }\n'}

def safe(model_key,case,seed,rep):
 name=f'six-{model_key}-{case}-s{seed}-r{rep}';path=OUT/(name+'.json')
 if path.exists():return json.loads(path.read_text())
 if spent()>8.0:return {'name':name,'error':'spend cap reached'}
 t=time.time();base={'name':name,'model_key':model_key,'model':MODELS[model_key],'case':case,'seed':seed,'rep':rep}
 try:
  data=run_case(MODELS[model_key],case);base.update(data);base['latency_s']=round(time.time()-t,2)
  base['totals']={k:sum(r.get('usage',{}).get(k,0) for r in [base.get('hidden'),*base.get('requests',[]),*base.get('dedup_requests',[]),*base.get('mcp_requests',[])] if r) for k in ['uncached','cached','cache_write','completion','calls']}
  base['totals']['total_tokens']=total_tokens(base['totals']);base['cost']=sum(((m.get('usage') or {}).get('cost') or {}).get('total') or 0 for r in [base.get('hidden'),*base.get('requests',[]),*base.get('dedup_requests',[])] if r for m in r.get('messages',[]))
 except Exception as e:
  base.update(error=str(e)[:1000],latency_s=round(time.time()-t,2))
 OUT.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
 with LOCK:
  with LEDGER.open('a') as f:f.write(json.dumps({'name':name,'cost':base.get('cost',0),'time':time.time()})+'\n')
 return base

def spent():
 if not LEDGER.exists():return 0
 return sum(json.loads(x).get('cost',0) for x in LEDGER.read_text().splitlines() if x.strip())
def seeds(text):
 out=[]
 for part in text.split(','):
  if '-' in part:
   a,b=map(int,part.split('-'));out.extend(range(a,b+1))
  else:out.append(int(part))
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',default=','.join(MODELS));ap.add_argument('--cases',default=','.join(CASES));ap.add_argument('--seeds',default='0');ap.add_argument('--reps',type=int,default=1);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
 jobs=[(m,c,s,r) for r in range(1,a.reps+1) for s in seeds(a.seeds) for m in a.models.split(',') for c in a.cases.split(',')]
 with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
  fs=[ex.submit(safe,*job) for job in jobs]
  for f in concurrent.futures.as_completed(fs):
   d=f.result();print(d['name'],'PASS' if d.get('pass') else ('ERR '+d.get('error','FAIL')),f"tokens={d.get('totals',{}).get('total_tokens')}",f"lat={d.get('latency_s')}",flush=True)
if __name__=='__main__':main()
