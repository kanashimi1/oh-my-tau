import json,sys
for line in sys.stdin:
 try:q=json.loads(line)
 except:continue
 method=q.get("method");rid=q.get("id")
 if rid is None:continue
 if method=="initialize":r={"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"six-fixture","version":"1"}}
 elif method=="tools/list":r={"tools":[{"name":"constant","description":"Return the fixture constant QZ-4197.","inputSchema":{"type":"object","properties":{},"additionalProperties":False}}]}
 elif method=="tools/call":r={"content":[{"type":"text","text":"QZ-4197"}],"isError":False}
 elif method=="ping":r={}
 else:continue
 print(json.dumps({"jsonrpc":"2.0","id":rid,"result":r}),flush=True)
