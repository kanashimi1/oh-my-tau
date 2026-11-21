"""Paired cost replay of real scoped tasks; fixtures come from immutable git revisions."""
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, tempfile, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=HERE/'results'/'real-replay'
import run as R
import longbench as L
from six_points import prompt

def load_manifest():return json.loads((OUT/'manifest.json').read_text())
def hashes(work):
 out={}
 for p in Path(work).rglob('*'):
  if p.is_file() and '.git' not in p.parts:
   out[str(p.relative_to(work))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return out

def materialize(task,dest):
 if dest.exists():return dest
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp')
 if tmp.exists():shutil.rmtree(tmp)
 tmp.mkdir(parents=True)
 archive=tempfile.NamedTemporaryFile(suffix='.tar',delete=False)
 try:
  with archive:
   subprocess.run(['git','archive','--output',archive.name,task['revision']],cwd=task['repo'],check=True)
  subprocess.run(['tar','-xf',archive.name,'-C',str(tmp)],check=True)
 finally:
  Path(archive.name).unlink(missing_ok=True)
 for rel in task['remove']:
  p=tmp/rel
  if p.is_dir():shutil.rmtree(p)
  elif p.exists():p.unlink()
 tmp.rename(dest);return dest

def run_one(task,model,arm,rep,tag,timeout):
 name=f'{tag}{task["id"]}-{arm}-r{rep}';target=OUT/f'{name}.json'
 if target.exists():raise FileExistsError(target)
 home=Path(tempfile.mkdtemp(prefix='omt-real-replay-'));work=home/'work';fixture=OUT/'fixtures'/task['id']
 shutil.copytree(fixture,work);before=hashes(work);agent,env=R.make_home(str(home))
 with open(Path(agent)/'config.yml','a') as f:f.write('modelRoles:\n  smol: zai/glm-5.3-flash\n')
 env.update(OMP_NO_UPDATE_CHECK='1',OMP_NO_TELEMETRY='1',OMP_CONTEXT_OFFLOAD_DIR=str(work/'.offload'),CARGO_TARGET_DIR=str(home/'cargo-target'))
 if arm=='rule':
  env.update(OMP_CONTEXT_POLICY='rule',OMP_CONTEXT_POLICY_RULE_OFFLOAD='8000',OMP_CONTEXT_POLICY_RULE_AGE='3',OMP_CONTEXT_POLICY_RULE_RESTATED='0',OMP_CONTEXT_POLICY_RULE_TOKENS='24000',OMP_CONTEXT_POLICY_RULE_EVERY='0')
 cmd=[str(ROOT/'target/shared/release/omt'),'--mode','rpc','--model',model,'--thinking','minimal','--tools','bash,read,write,edit,eval,lsp','--no-extensions','--no-skills','--no-rules','--no-title','--no-lsp','--no-pty','--context-strategy','clm' if arm=='rule' else 'append']
 rpc=R.Rpc(cmd,env,str(work));t0=time.time()
 error=None
 try:
  text=task['prompt'].replace(task['original_cwd'],str(work))
  text+='\n\nSandbox rule: work only in the current directory. Do not access the original repository. Complete the task and run its verification.'
  result=prompt(rpc,text,timeout)
 except Exception as exc:
  error=f'{type(exc).__name__}: {exc}'
  messages=[]
  for session in (home/'.omp-rs/sessions').glob('*/*.jsonl'):
   for line in session.open(errors='ignore'):
    try:event=json.loads(line)
    except json.JSONDecodeError:continue
    if event.get('type')=='message' and isinstance(event.get('message'),dict):messages.append(event['message'])
  result={'status':'error','messages':messages,'usage':{},'tools':[],'reply':''}
 finally:rpc.close()
 after=hashes(work);ignored=set(task['allowed'])|set(task.get('generated',[]))
 def allowed(rel):return rel in ignored or any(rel.startswith(x.rstrip('/')+'/') for x in task.get('generated',[])) or rel.startswith('.offload/')
 mutations=[rel for rel in set(before)|set(after) if not allowed(rel) and before.get(rel)!=after.get(rel)]
 try:check=subprocess.run(['bash','-lc',task['check']],cwd=work,env={**env,'CARGO_TARGET_DIR':str(home/'cargo-target')},capture_output=True,text=True,timeout=timeout)
 except subprocess.TimeoutExpired as exc:
  check=subprocess.CompletedProcess(exc.cmd,124,(exc.stdout or b'').decode() if isinstance(exc.stdout,bytes) else exc.stdout or '',(exc.stderr or b'').decode() if isinstance(exc.stderr,bytes) else exc.stderr or '')
 sub_cost=0;sub_calls=0
 for p in (home/'.omp-rs/sessions').rglob('*.jsonl'):
  for line in p.open(errors='ignore'):
   if '"model_usage"' not in line:continue
   try:d=json.loads(line)
   except:continue
   if d.get('type')=='model_usage' and d.get('purpose')=='rlm':
    sub_calls+=1;sub_cost+=((d.get('usage') or {}).get('cost') or {}).get('total') or 0
 root_cost=L.cost_of([{'usage':m['usage']} for m in result['messages'] if m.get('usage')])
 success=result['status']=='completed' and check.returncode==0 and not mutations
 data={'name':name,'task':task['id'],'model':model,'arm':arm,'rep':rep,'source_session':task['session'],'source_revision':task['revision'],'status':result['status'],'success':success,'check_returncode':check.returncode,'check_tail':(check.stdout+check.stderr)[-4000:],'source_mutations':mutations,'cost':root_cost+sub_cost,'root_cost':root_cost,'sub_cost':sub_cost,'sub_calls':sub_calls,'usage':result['usage'],'tool_calls':len(result['tools']),'latency_s':round(time.time()-t0,2),'error':error,'reply':result['reply'],'messages':result['messages'],'tools':result['tools']}
 target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');shutil.rmtree(home,ignore_errors=True);return data

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--model');ap.add_argument('--arms',default='append,rule');ap.add_argument('--tasks',default='all');ap.add_argument('--reps',type=int,default=1);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--tag',default='real-');ap.add_argument('--timeout',type=int,default=1800);a=ap.parse_args()
 manifest=load_manifest();tasks=manifest['tasks']
 if a.tasks!='all':
  wanted=set(a.tasks.split(','));tasks=[t for t in manifest['tasks'] if t['id'] in wanted]
 for t in tasks:materialize(t,OUT/'fixtures'/t['id'])
 if a.prepare_only:
  for t in tasks:print(t['id'],OUT/'fixtures'/t['id'])
  return
 if not a.model:ap.error('--model is required unless --prepare-only')
 jobs=[(t,a.model,arm,r,a.tag,a.timeout) for t in tasks for arm in a.arms.split(',') for r in range(1,a.reps+1)]
 jobs=[j for j in jobs if not (OUT/f'{a.tag}{j[0]["id"]}-{j[2]}-r{j[3]}.json').exists()]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  futures=[ex.submit(run_one,*j) for j in jobs]
  for future in concurrent.futures.as_completed(futures):
   try:d=future.result()
   except Exception as exc:print('replay infrastructure error:',type(exc).__name__,str(exc),flush=True);continue
   print(d['name'],'success=',d['success'],'cost=',round(d['cost'],6),'latency=',d['latency_s'],flush=True)
if __name__=='__main__':main()
