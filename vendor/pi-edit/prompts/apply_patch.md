Edit files via the `apply_patch` shell command: a stripped-down, file-oriented diff, easy to parse and safe to apply.

Envelope:
```
*** Begin Patch
[ one or more file sections ]
*** End Patch
```
Each file operation MUST start with an action header:
`*** Add File: <path>`: create file; every following line MUST start with `+` (initial content).
`*** Delete File: <path>`: remove existing file; nothing follows.
`*** Update File: <path>`: patch in place; optional immediate `*** Move to: <new path>` renames; then one or more `@@` hunks (header optional). Hunk lines start with space, `-` or `+`.

Context: 3 unchanged lines before and after each change. For changes within 3 lines of each other, do NOT repeat the first change's after-context as the second's before-context. If 3 lines don't identify the code uniquely, add `@@ <class/function>`; if still ambiguous, stack several `@@` lines:
```
@@ class BaseClass
@@ 	 def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]
```

Grammar:
```
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE
```

Example combining operations:
```
*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch
```

Paths MUST be relative, NEVER absolute.
