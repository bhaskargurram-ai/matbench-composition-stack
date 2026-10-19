# source this: helper for driving idea-spark on Windows (python3 -> .venv python)
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
P=/c/Users/Bhaskar/Downloads/Matbench_Paper
PY="$P/.venv/Scripts/python.exe"
SK="$P/.claude/skills/idea-spark"
RD='C:\Users\Bhaskar\Downloads\Matbench_Paper\ideaspark_run\matbench-composition-stack'
python3() { "$PY" "$@"; }
nxt() { "$PY" "$SK/scripts/run.py" next --dir "$RD" 2>&1 | head -c 6000; }
runstep() { local cmd; cmd=$("$PY" "$SK/scripts/run.py" next --dir "$RD" 2>&1 | grep '^RUN    : ' | sed 's/^RUN    : //'); echo ">> $cmd" | cut -c1-300; eval "$cmd"; }
# advance through bash steps; stop at an llm step (written to current_step.txt) or terminal state
advance() { local i out typ; for i in $(seq 1 12); do out=$("$PY" "$SK/scripts/run.py" next --dir "$RD" 2>&1); typ=$(echo "$out" | grep '^TYPE' | head -1); echo "$(echo "$out" | grep '^STATE\|^STEP' | tr '\n' ' ')" | cut -c1-200; if echo "$typ" | grep -q bash; then runstep 2>&1 | grep -v SyntaxWarning | grep -v '^  `' | tail -2; else echo "$out" > "$P/ideaspark_run/current_step.txt"; return; fi; done; }
