"""
BugLens Flask API — Multi-Language + AI Suggest
=================================================
Run: python app.py
API: http://localhost:5001

Endpoints:
  POST /analyze   { "code": "...", "lang": "java" }  -> bug analysis
  POST /suggest   { "code": "...", "lang": "java" }  -> AI-powered suggestions
  GET  /health

Supported Languages: python, java, c, cpp, javascript, html
"""

import os, re, ast, sys
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════════════
# PYTHON-SPECIFIC: Layer 1 — Syntax via compile()
# ══════════════════════════════════════════════════════════════════════

def check_python_syntax(code):
    bugs = []
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        ln      = e.lineno or 1
        lines   = code.split("\n")
        snippet = lines[ln-1].strip() if 0 < ln <= len(lines) else ""
        msg     = (e.msg or "").lower()
        BLOCK   = ["def ","class ","if ","elif ","else","for ","while ","with ","try","except","finally"]
        if "expected ':'" in msg or ("invalid syntax" in msg and any(k in snippet for k in BLOCK)):
            name,desc = "Missing Colon","Block statement missing ':' at end of line."
            fixes = [{"label":"Fix — Add ':'","approach":"Add colon at end of block line.",
                      "code": snippet.rstrip() + ":"}]
        elif "indent" in msg:
            name,desc = "Indentation Error","Inconsistent indentation. Use exactly 4 spaces per level."
            fixes = [{"label":"Fix — 4-space indent","approach":"Use 4 spaces, never tabs.",
                      "code":"def f():\n    pass  # 4 spaces"}]
        elif "eof" in msg or "never closed" in msg:
            name,desc = "Unclosed Bracket","Opening bracket/paren/brace never closed."
            fixes = [{"label":"Fix — Close bracket","approach":"Add matching closing bracket.",
                      "code":"result = func(a, b)  # closed\ndata = [1, 2, 3]     # closed"}]
        elif "unterminated" in msg or "eol" in msg:
            name,desc = "Unterminated String","String literal opened but never closed."
            fixes = [{"label":"Fix — Close string","approach":"Add matching closing quote.",
                      "code":"msg = 'hello'   # single\nmsg = \"hello\"  # double"}]
        elif "invalid escape" in msg:
            name,desc = "Invalid Escape Sequence","Backslash sequence not valid outside raw string."
            fixes = [{"label":"Fix A — Raw string","approach":"Prefix with r.",
                      "code":"path = r'C:\\Users\\pawan'"},
                     {"label":"Fix B — Double backslash","approach":"Escape each backslash.",
                      "code":"path = 'C:\\\\Users\\\\pawan'"}]
        else:
            name,desc = "Syntax Error", e.msg or "Python cannot parse this code."
            fixes = [{"label":"Fix — Review line","approach":"Check for typos or missing keywords.",
                      "code":f"# Line {ln}: {snippet}"}]
        bugs.append({"id":"SYNTAX_ERROR","name":name,"severity":"CRITICAL",
                     "line":ln,"lineSnippet":snippet,"desc":desc,"fixes":fixes})
    except IndentationError as e:
        ln      = e.lineno or 1
        lines   = code.split("\n")
        snippet = lines[ln-1].strip() if 0 < ln <= len(lines) else ""
        bugs.append({"id":"INDENT_ERROR","name":"Indentation Error","severity":"CRITICAL",
                     "line":ln,"lineSnippet":snippet,
                     "desc":"Unexpected indentation. Use consistent 4-space blocks.",
                     "fixes":[{"label":"Fix","approach":"4 spaces per level.",
                                "code":"def f():\n    pass"}]})
    return bugs


# ══════════════════════════════════════════════════════════════════════
# PYTHON-SPECIFIC: Layer 2 — AST semantic checks
# ══════════════════════════════════════════════════════════════════════

def check_python_ast(code):
    bugs  = []
    lines = code.split("\n")
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    for node in ast.walk(tree):

        # Division by zero
        if isinstance(node, ast.BinOp) and isinstance(node.op,(ast.Div,ast.Mod,ast.FloorDiv)):
            if isinstance(node.right,ast.Constant) and node.right.value == 0:
                ln = node.lineno
                bugs.append({"id":"DIV_ZERO","name":"Division by Zero","severity":"CRITICAL",
                    "line":ln,"lineSnippet":lines[ln-1].strip() if ln<=len(lines) else "",
                    "desc":"Dividing by 0 always raises ZeroDivisionError.",
                    "fixes":[
                        {"label":"Fix A — Guard","approach":"Check before dividing.",
                         "code":"if divisor != 0:\n    result = a / divisor"},
                        {"label":"Fix B — try/except","approach":"Catch ZeroDivisionError.",
                         "code":"try:\n    r = a/b\nexcept ZeroDivisionError:\n    r = 0"},
                    ]})

        # Bare except
        if isinstance(node,ast.ExceptHandler) and node.type is None:
            ln = node.lineno
            bugs.append({"id":"BARE_EXCEPT","name":"Bare Exception Catch","severity":"MEDIUM",
                "line":ln,"lineSnippet":lines[ln-1].strip() if ln<=len(lines) else "",
                "desc":"'except:' catches everything including KeyboardInterrupt.",
                "fixes":[
                    {"label":"Fix A — Specific","approach":"Name the exception.",
                     "code":"except ValueError as e:\n    print(e)"},
                    {"label":"Fix B — Base Exception","approach":"Catch all non-system.",
                     "code":"except Exception as e:\n    logging.error(e)"},
                ]})

        # Mutable default
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for default in node.args.defaults:
                if isinstance(default,(ast.List,ast.Dict,ast.Set)):
                    ln = node.lineno
                    t  = {ast.List:"list",ast.Dict:"dict",ast.Set:"set"}.get(type(default),"mutable")
                    bugs.append({"id":"MUTABLE_DEFAULT","name":"Mutable Default Argument","severity":"HIGH",
                        "line":ln,"lineSnippet":lines[ln-1].strip() if ln<=len(lines) else "",
                        "desc":f"Default {t} shared across all calls — mutations persist.",
                        "fixes":[{"label":"Fix — None sentinel","approach":"Use None, create inside.",
                                  "code":"def func(items=None):\n    if items is None: items=[]\n    items.append(1)\n    return items"}]})

        # == None
        if isinstance(node,ast.Compare):
            for op,comp in zip(node.ops,node.comparators):
                if isinstance(op,(ast.Eq,ast.NotEq)) and isinstance(comp,ast.Constant) and comp.value is None:
                    ln  = node.lineno
                    old = "==" if isinstance(op,ast.Eq) else "!="
                    new = "is" if isinstance(op,ast.Eq) else "is not"
                    bugs.append({"id":"CMP_NONE","name":f"Use '{new}' for None","severity":"LOW",
                        "line":ln,"lineSnippet":lines[ln-1].strip() if ln<=len(lines) else "",
                        "desc":f"PEP8: use '{new} None' not '{old} None'.",
                        "fixes":[{"label":f"Fix — '{new}'","approach":f"Replace '{old} None'.",
                                  "code":f"if x {new} None:  # correct"}]})

        # Infinite loop
        if isinstance(node,ast.While) and isinstance(node.test,ast.Constant) and node.test.value:
            has_exit = any(isinstance(n,(ast.Break,ast.Return))
                           for n in ast.walk(ast.Module(body=node.body,type_ignores=[])))
            if not has_exit:
                ln = node.lineno
                bugs.append({"id":"INF_LOOP","name":"Infinite Loop — No Exit","severity":"HIGH",
                    "line":ln,"lineSnippet":lines[ln-1].strip() if ln<=len(lines) else "",
                    "desc":"while True with no break/return runs forever.",
                    "fixes":[
                        {"label":"Fix A — break","approach":"Break on condition.",
                         "code":"while True:\n    if done: break"},
                        {"label":"Fix B — condition","approach":"Use real loop condition.",
                         "code":"while not_done:\n    not_done = step()"},
                    ]})

        # Missing return on all paths
        if isinstance(node,ast.FunctionDef):
            has_val = any(isinstance(n,ast.Return) and n.value is not None for n in ast.walk(node))
            last_ret = bool(node.body) and isinstance(node.body[-1],ast.Return)
            if has_val and not last_ret:
                ln = node.lineno
                bugs.append({"id":"MISSING_RETURN","name":"Not All Paths Return","severity":"HIGH",
                    "line":ln,"lineSnippet":lines[ln-1].strip() if ln<=len(lines) else "",
                    "desc":"Function may fall off end, returning None implicitly.",
                    "fixes":[
                        {"label":"Fix A — explicit return","approach":"Return at every path.",
                         "code":"def f(x):\n    if x>0: return x*2\n    return 0"},
                        {"label":"Fix B — raise","approach":"Raise on invalid path.",
                         "code":"def f(x):\n    if x>0: return x*2\n    raise ValueError(x)"},
                    ]})

    return bugs


# ══════════════════════════════════════════════════════════════════════
# JAVA: Full error detection
# ══════════════════════════════════════════════════════════════════════

def check_java(code):
    bugs  = []
    lines = code.split("\n")

    def add(id_, name, sev, ln, snippet, desc, fixes):
        bugs.append({"id":id_,"name":name,"severity":sev,
                     "line":ln,"lineSnippet":snippet,"desc":desc,"fixes":fixes})

    for i, raw in enumerate(lines):
        ln  = i + 1
        s   = raw.strip()

        # Missing semicolon — statement lines not ending with ; { } // or being blank
        if (s and not s.startswith("//") and not s.startswith("*") and not s.startswith("/*")
                and not s.endswith(("{","}",";")) and not s.startswith("@")
                and not s.startswith("import") and not s.startswith("package")
                and re.search(r"^\s*(int|String|double|float|boolean|char|long|byte|short|var"
                               r"|System\.|return |throw |break|continue"
                               r"|\w+\s*[\+\-\*\/]?=)", raw)
                and not re.search(r"(if|else|for|while|try|catch|finally|class|interface|enum"
                                   r"|public|private|protected|static|void|new)\s*[\(\{]?$", s)):
            add("MISSING_SEMI","Missing Semicolon","CRITICAL",ln,s,
                "Java statement must end with a semicolon ';'.",
                [{"label":"Fix — Add semicolon","approach":"Add ';' at the end of the statement.",
                  "code": s.rstrip(";") + ";"}])

        # Wrong import syntax
        if re.match(r"import\s+java\.\w+;?$", s) and "." not in s.split("import")[1].strip().rstrip(";"):
            add("BAD_IMPORT","Incomplete Import Statement","CRITICAL",ln,s,
                "Java import must specify the full package path e.g. java.util.Scanner.",
                [{"label":"Fix — Full import","approach":"Use the full qualified import path.",
                  "code":"import java.io.*;          // all io classes\nimport java.util.Scanner; // specific class"}])

        # import java.io without asterisk or class name
        if re.match(r"^import\s+java\.io\s*;", s):
            add("BAD_IMPORT","Invalid Import: java.io","CRITICAL",ln,s,
                "'import java.io;' is invalid. Must be 'import java.io.*' or a specific class.",
                [{"label":"Fix A — Wildcard","approach":"Import all io classes.",
                  "code":"import java.io.*;"},
                 {"label":"Fix B — Specific","approach":"Import only what you need.",
                  "code":"import java.io.IOException;\nimport java.io.FileReader;"}])

        # Wrong println case
        if "System.out.Println" in raw or "System.out.PrintLn" in raw or "System.out.PRINTLN" in raw:
            add("WRONG_CASE","Wrong Method Case: Println","CRITICAL",ln,s,
                "Java is case-sensitive. 'Println' does not exist — use 'println' (lowercase p).",
                [{"label":"Fix — Lowercase","approach":"Change to System.out.println().",
                  "code":raw.replace("Println","println").replace("PrintLn","println").strip()}])

        # System.out.print without ln and without semicolon check
        if re.search(r"System\.out\.(print|println)\s*\([^)]*\)\s*$", s) and not s.endswith(";"):
            add("MISSING_SEMI_PRINT","Missing Semicolon After println","CRITICAL",ln,s,
                "println() call is missing the semicolon.",
                [{"label":"Fix","approach":"Add semicolon.","code":s+";"}])

        # Comparing int to String with ==
        if re.search(r'\b\w+\s*==\s*"', raw) or re.search(r'"\s*==\s*\w+', raw):
            # Check if left side is likely a number/variable (not String declared)
            if not re.search(r'String\s+\w+\s*=.*==', raw):
                add("TYPE_MISMATCH","Type Mismatch: int/variable compared to String","CRITICAL",ln,s,
                    "Cannot compare a non-String value to a String literal with ==. "
                    "For String comparison use .equals().",
                    [{"label":"Fix A — .equals()","approach":"Use .equals() for String comparison.",
                      "code":'if (str.equals("10")) { ... }'},
                     {"label":"Fix B — parseInt","approach":"Parse string to int for numeric compare.",
                      "code":"if (number == Integer.parseInt(input)) { ... }"},
                     {"label":"Fix C — String.valueOf","approach":"Convert int to String.",
                      "code":'if (String.valueOf(number).equals("10")) { ... }'}])

        # Undeclared loop variable i
        if re.search(r"for\s*\(\s*i\s*=", raw):
            # Check if i was declared above
            declared_above = any(re.search(r"\bint\s+i\b", lines[j]) for j in range(max(0,i-10), i))
            if not declared_above:
                add("UNDECLARED_VAR","Undeclared Variable 'i' in for-loop","CRITICAL",ln,s,
                    "Variable 'i' used in for-loop but never declared. Must declare type.",
                    [{"label":"Fix — Declare in loop","approach":"Declare int i inside the for statement.",
                      "code":"for (int i = 0; i < 5; i++) { ... }"}])

        # Uninitialized variable used in +=
        if re.search(r"\w+\s*\+=", raw):
            var = re.search(r"(\w+)\s*\+=", raw)
            if var:
                vname = var.group(1)
                declared = any(re.search(rf"\b(int|long|double|float)\s+{vname}\s*;", lines[j])
                                for j in range(max(0,i-20), i))
                initialized = any(re.search(rf"\b(int|long|double|float)\s+{vname}\s*=", lines[j])
                                   for j in range(max(0,i-20), i))
                if declared and not initialized:
                    add("UNINIT_VAR",f"Uninitialized Variable '{vname}'","HIGH",ln,s,
                        f"'{vname}' was declared but never initialized. In Java, local variables "
                        "must be explicitly initialized before use.",
                        [{"label":"Fix — Initialize to 0","approach":"Set initial value in declaration.",
                          "code":f"int {vname} = 0;  // initialized"},
                         {"label":"Fix B — Initialize and declare","approach":"Declare with value.",
                          "code":f"int {vname} = 0; // then use {vname} += ..."}])

        # Undefined variable (common name mismatches like result vs results)
        if re.search(r'\bSystem\.out\.println\s*\(\s*"[^"]*"\s*\+\s*\w+', raw):
            var_used = re.search(r'println\s*\([^+]*\+\s*(\w+)', raw)
            if var_used:
                vname = var_used.group(1)
                declared = any(re.search(rf"\b\w+\s+{vname}\b", lines[j])
                                for j in range(max(0,i-30), i))
                if not declared:
                    add("UNDEF_VAR",f"Undefined Variable '{vname}'","CRITICAL",ln,s,
                        f"Variable '{vname}' is used but was never declared. "
                        "Check for typos in variable names.",
                        [{"label":f"Fix — Declare '{vname}'","approach":"Declare the variable before use.",
                          "code":f"int {vname} = 0; // declare first\n// OR rename to match existing variable"},
                         {"label":"Fix — Check spelling","approach":"You may have a typo.",
                          "code":"// e.g. 'result' vs 'results' — pick one and use consistently"}])

        # ArithmeticException risk — division by user input
        if re.search(r"\w+\s*/\s*(divisor|denominator|n|d|b)\b", raw):
            add("ARITH_EXCEPTION","Potential ArithmeticException (÷0)","HIGH",ln,s,
                "Division by a user-supplied variable — crashes with ArithmeticException if value is 0.",
                [{"label":"Fix A — Guard check","approach":"Check before dividing.",
                  "code":"if (divisor != 0) {\n    int result = number / divisor;\n} else {\n    System.out.println(\"Cannot divide by zero\");\n}"},
                 {"label":"Fix B — try/catch","approach":"Catch ArithmeticException at runtime.",
                  "code":"try {\n    int result = number / divisor;\n} catch (ArithmeticException e) {\n    System.out.println(\"Error: \" + e.getMessage());\n}"}])

        # NullPointerException risk
        if re.search(r"\w+\.length\(\)|\.charAt\(|\.substring\(|\.equals\(", raw):
            obj = re.search(r"(\w+)\.\w+\(", raw)
            if obj and obj.group(1) not in ("System","Math","String","Arrays","Collections"):
                add("NPE_RISK","Potential NullPointerException","HIGH",ln,s,
                    f"Method called on '{obj.group(1)}' which could be null if initialization failed.",
                    [{"label":"Fix A — Null check","approach":"Check for null before calling methods.",
                      "code":f"if ({obj.group(1)} != null) {{\n    {obj.group(1)}.yourMethod();\n}}"},
                     {"label":"Fix B — Optional","approach":"Use Optional for nullable values (Java 8+).",
                      "code":f"Optional.ofNullable({obj.group(1)})\n         .ifPresent(v -> v.yourMethod());"}])

        # Resource leak — Scanner/FileReader not closed
        if re.search(r"new\s+Scanner\s*\(|new\s+FileReader\s*\(|new\s+BufferedReader\s*\(", raw):
            closed_below = any("close()" in lines[j] or "try-with-resources" in lines[j]
                                for j in range(i, min(len(lines), i+30)))
            if not closed_below:
                add("RESOURCE_LEAK","Resource Leak — not closed","HIGH",ln,s,
                    "Scanner/Reader opened but close() not found. Use try-with-resources.",
                    [{"label":"Fix A — try-with-resources","approach":"Auto-closes on exit (Java 7+).",
                      "code":"try (Scanner scanner = new Scanner(System.in)) {\n    int n = scanner.nextInt();\n}  // auto-closed"},
                     {"label":"Fix B — finally close","approach":"Explicitly close in finally.",
                      "code":"Scanner sc = new Scanner(System.in);\ntry {\n    int n = sc.nextInt();\n} finally {\n    sc.close();\n}"}])

        # Array index out of bounds risk
        if re.search(r"\w+\[[\w\+\-]+\]", raw) and not re.search(r"(int|String|double)\s*\[", raw):
            add("ARRAY_OOB","Potential ArrayIndexOutOfBoundsException","MEDIUM",ln,s,
                "Array accessed by index without bounds check.",
                [{"label":"Fix A — Bounds check","approach":"Verify index before accessing.",
                  "code":"if (i >= 0 && i < arr.length) {\n    val = arr[i];\n}"},
                 {"label":"Fix B — try/catch","approach":"Catch ArrayIndexOutOfBoundsException.",
                  "code":"try {\n    val = arr[i];\n} catch (ArrayIndexOutOfBoundsException e) {\n    System.out.println(\"Index out of range\");\n}"}])

        # ClassCastException risk
        if re.search(r"\(\s*(Integer|String|Double|Long)\s*\)\s*\w+", raw):
            add("CLASS_CAST","Potential ClassCastException","MEDIUM",ln,s,
                "Explicit cast detected. If the object is not the correct type at runtime, ClassCastException is thrown.",
                [{"label":"Fix A — instanceof check","approach":"Verify type before casting.",
                  "code":"if (obj instanceof Integer) {\n    int n = (Integer) obj;\n}"},
                 {"label":"Fix B — Generics","approach":"Use generics to avoid runtime casts.",
                  "code":"List<Integer> list = new ArrayList<>();  // type-safe"}])

    return bugs


# ══════════════════════════════════════════════════════════════════════
# JAVASCRIPT: Error detection
# ══════════════════════════════════════════════════════════════════════

def check_javascript(code):
    bugs  = []
    lines = code.split("\n")

    for i, raw in enumerate(lines):
        ln = i + 1
        s  = raw.strip()

        # == instead of ===
        if re.search(r"[^=!<>]==[^=]", raw) and "==" in raw:
            bugs.append({"id":"LOOSE_EQUAL","name":"Loose Equality (==) — Use ===","severity":"HIGH",
                "line":ln,"lineSnippet":s,
                "desc":"== performs type coercion. '5' == 5 is true in JS. Use === for strict comparison.",
                "fixes":[
                    {"label":"Fix — Use ===","approach":"Replace == with === for strict equality.",
                     "code":raw.replace("==","===").replace("!===","!==").strip()},
                ]})

        # var instead of let/const
        if re.match(r"^\s*var\s+", raw):
            bugs.append({"id":"VAR_USAGE","name":"Use 'let' or 'const' Instead of 'var'","severity":"MEDIUM",
                "line":ln,"lineSnippet":s,
                "desc":"'var' is function-scoped and hoisted, causing subtle bugs. Use 'let' or 'const'.",
                "fixes":[
                    {"label":"Fix A — const","approach":"Use const for values that don't change.",
                     "code":raw.replace("var ","const ",1).strip()},
                    {"label":"Fix B — let","approach":"Use let for values that change.",
                     "code":raw.replace("var ","let ",1).strip()},
                ]})

        # undefined variable check (common pattern)
        if re.search(r"\bconsole\.log\s*\(", raw) and not s.endswith(";"):
            bugs.append({"id":"MISSING_SEMI_JS","name":"Missing Semicolon","severity":"LOW",
                "line":ln,"lineSnippet":s,
                "desc":"JavaScript statement missing semicolon (ASI can sometimes fail).",
                "fixes":[{"label":"Fix — Add ;","approach":"Add semicolon at end.",
                          "code":s+";"}]})

        # == null instead of === null
        if re.search(r"==\s*null\b|null\s*==", raw):
            bugs.append({"id":"NULL_CHECK_JS","name":"Use === null for Null Check","severity":"MEDIUM",
                "line":ln,"lineSnippet":s,
                "desc":"== null matches both null and undefined. Use === null if you want only null.",
                "fixes":[
                    {"label":"Fix A — === null","approach":"Strict null check.",
                     "code":raw.replace("== null","=== null").strip()},
                    {"label":"Fix B — nullish coalescing","approach":"Use ?? for null/undefined default.",
                     "code":"const val = input ?? 'default';  // handles null and undefined"},
                ]})

        # eval usage
        if re.search(r"\beval\s*\(", raw):
            bugs.append({"id":"EVAL_JS","name":"Dangerous eval()","severity":"CRITICAL",
                "line":ln,"lineSnippet":s,
                "desc":"eval() executes arbitrary JavaScript — XSS/RCE risk.",
                "fixes":[
                    {"label":"Fix A — JSON.parse","approach":"For data parsing.",
                     "code":"const data = JSON.parse(input);"},
                    {"label":"Fix B — Function map","approach":"Map strings to allowed functions.",
                     "code":"const ops = { add: (a,b)=>a+b };\nconst result = ops[op](x, y);"},
                ]})

        # Async without await
        if re.search(r"async\s+function|async\s+\(", raw):
            if "await" not in code:
                bugs.append({"id":"ASYNC_NO_AWAIT","name":"async Function With No await","severity":"MEDIUM",
                    "line":ln,"lineSnippet":s,
                    "desc":"Function marked async but no await found — likely returns unwrapped Promise.",
                    "fixes":[
                        {"label":"Fix — Add await","approach":"await the async operations inside.",
                         "code":"async function fetchData() {\n    const res = await fetch(url);\n    const data = await res.json();\n    return data;\n}"},
                    ]})

    return bugs


# ══════════════════════════════════════════════════════════════════════
# C / C++: Error detection
# ══════════════════════════════════════════════════════════════════════

def check_c_cpp(code, lang="c"):
    bugs  = []
    lines = code.split("\n")
    is_cpp = (lang == "cpp")

    for i, raw in enumerate(lines):
        ln = i + 1
        s  = raw.strip()

        # Missing semicolon
        if (s and not s.startswith("//") and not s.startswith("#") and not s.startswith("/*")
                and not s.endswith(("{","}",";","\\"))
                and re.search(r"^\s*(int|char|float|double|long|short|unsigned|bool|auto"
                               r"|return |printf|scanf|cout|cin|\w+\s*[\+\-\*\/]?=)", raw)
                and not re.search(r"(if|else|for|while|do|switch|struct|class|enum)\s*[\(\{]?$", s)):
            bugs.append({"id":"MISSING_SEMI_C","name":"Missing Semicolon","severity":"CRITICAL",
                "line":ln,"lineSnippet":s,"desc":"C/C++ statement must end with ';'.",
                "fixes":[{"label":"Fix — Add ;","approach":"Add semicolon at end of statement.",
                          "code":s.rstrip(";")+";"}]})

        # NULL pointer dereference
        if re.search(r"malloc\s*\(|calloc\s*\(", raw):
            check_null = any("if" in lines[j] and ("NULL" in lines[j] or "null" in lines[j])
                              for j in range(i, min(len(lines),i+3)))
            if not check_null:
                bugs.append({"id":"NULL_PTR","name":"Potential NULL Pointer (malloc without check)","severity":"CRITICAL",
                    "line":ln,"lineSnippet":s,
                    "desc":"malloc/calloc can return NULL if allocation fails. Dereferencing NULL causes segfault.",
                    "fixes":[
                        {"label":"Fix — Check NULL","approach":"Always check malloc return value.",
                         "code":"int *ptr = malloc(sizeof(int) * n);\nif (ptr == NULL) {\n    fprintf(stderr, \"Memory allocation failed\\n\");\n    exit(1);\n}"},
                    ]})

        # Buffer overflow risk
        if re.search(r"\bgets\s*\(", raw):
            bugs.append({"id":"BUFFER_OVERFLOW","name":"Buffer Overflow Risk: gets()","severity":"CRITICAL",
                "line":ln,"lineSnippet":s,
                "desc":"gets() has no bounds checking — classic buffer overflow vulnerability (CWE-120).",
                "fixes":[
                    {"label":"Fix A — fgets","approach":"Use fgets with buffer size.",
                     "code":"fgets(buffer, sizeof(buffer), stdin);  // safe"},
                    {"label":"Fix B — scanf with width","approach":"Limit input with scanf.",
                     "code":'scanf("%99s", buffer);  // max 99 chars'},
                ]})

        # strcpy without bounds
        if re.search(r"\bstrcpy\s*\(", raw):
            bugs.append({"id":"STRCPY_UNSAFE","name":"Unsafe strcpy — Use strncpy","severity":"HIGH",
                "line":ln,"lineSnippet":s,
                "desc":"strcpy does not check destination buffer size — buffer overflow risk.",
                "fixes":[
                    {"label":"Fix A — strncpy","approach":"Limit copy length.",
                     "code":"strncpy(dest, src, sizeof(dest)-1);\ndest[sizeof(dest)-1] = '\\0';  // ensure null-terminated"},
                    {"label":"Fix B — snprintf","approach":"Even safer for formatted copies.",
                     "code":"snprintf(dest, sizeof(dest), \"%s\", src);"},
                ]})

        # Use after free
        if re.search(r"\bfree\s*\((\w+)\)", raw):
            freed = re.search(r"\bfree\s*\((\w+)\)", raw).group(1)
            used_after = any(re.search(rf"\b{freed}\b", lines[j])
                              for j in range(i+1, min(len(lines),i+10)))
            if used_after:
                bugs.append({"id":"USE_AFTER_FREE","name":"Use After Free","severity":"CRITICAL",
                    "line":ln,"lineSnippet":s,
                    "desc":f"'{freed}' is freed on this line but may be accessed afterward — undefined behavior.",
                    "fixes":[
                        {"label":"Fix — Set NULL after free","approach":"Null the pointer after freeing.",
                         "code":f"free({freed});\n{freed} = NULL;  // prevents use-after-free"},
                    ]})

        # Division without zero check
        if re.search(r"/\s*\w+\b", raw) and not re.search(r"//", raw):
            divisor = re.search(r"/\s*(\w+)", raw)
            if divisor and divisor.group(1) not in ("2","n","sizeof","strlen","1"):
                bugs.append({"id":"DIV_ZERO_C","name":"Potential Division by Zero","severity":"HIGH",
                    "line":ln,"lineSnippet":s,
                    "desc":"Division by variable without zero check may cause SIGFPE crash.",
                    "fixes":[
                        {"label":"Fix — Check before divide","approach":"Guard the division.",
                         "code":f"if ({divisor.group(1)} != 0) {{\n    result = a / {divisor.group(1)};\n}}"},
                    ]})

        # C++ specific
        if is_cpp:
            if re.search(r"using namespace std\s*;", raw):
                bugs.append({"id":"USING_STD","name":"'using namespace std' in Header/Global","severity":"LOW",
                    "line":ln,"lineSnippet":s,
                    "desc":"'using namespace std' globally causes name collisions in large projects.",
                    "fixes":[
                        {"label":"Fix A — Explicit std::","approach":"Use std:: prefix explicitly.",
                         "code":"std::cout << \"hello\" << std::endl;"},
                        {"label":"Fix B — Scoped using","approach":"Limit scope to function.",
                         "code":"void func() {\n    using namespace std;\n    cout << \"hello\";\n}"},
                    ]})

    return bugs


# ══════════════════════════════════════════════════════════════════════
# HTML: Error detection
# ══════════════════════════════════════════════════════════════════════

def check_html(code):
    bugs  = []
    lines = code.split("\n")

    for i, raw in enumerate(lines):
        ln = i + 1
        s  = raw.strip()

        # Unclosed tags (basic check)
        tags = re.findall(r"<(\w+)[^>]*>", raw)
        close = re.findall(r"</(\w+)>", raw)
        SELF_CLOSING = {"br","hr","img","input","meta","link","area","base","col","embed","param","source","track","wbr"}
        for tag in tags:
            if tag.lower() not in SELF_CLOSING and tag.lower() not in close:
                # check if closed on next few lines
                closed_near = any(f"</{tag}" in lines[j] for j in range(i, min(len(lines),i+10)))
                if not closed_near:
                    bugs.append({"id":"UNCLOSED_TAG",f"name":f"Unclosed <{tag}> Tag","severity":"HIGH",
                        "line":ln,"lineSnippet":s,
                        "desc":f"<{tag}> opened but matching </{tag}> not found nearby.",
                        "fixes":[{"label":f"Fix — Close <{tag}>","approach":f"Add </{tag}> at appropriate position.",
                                  "code":f"<{tag}>content</{tag}>"}]})

        # Missing alt on img
        if re.search(r"<img\b(?![^>]*\balt=)", raw, re.IGNORECASE):
            bugs.append({"id":"IMG_NO_ALT","name":"<img> Missing alt Attribute","severity":"MEDIUM",
                "line":ln,"lineSnippet":s,
                "desc":"Images must have alt text for accessibility (WCAG 2.1).",
                "fixes":[{"label":"Fix — Add alt","approach":"Add descriptive alt text.",
                          "code":'<img src="image.jpg" alt="Description of image">'}]})

        # Inline styles (best practice warning)
        if re.search(r'\bstyle\s*=\s*"', raw, re.IGNORECASE):
            bugs.append({"id":"INLINE_STYLE","name":"Inline Style — Use CSS Class","severity":"LOW",
                "line":ln,"lineSnippet":s,
                "desc":"Inline styles are hard to maintain and override. Use CSS classes.",
                "fixes":[{"label":"Fix — CSS class","approach":"Move styles to external CSS.",
                          "code":"/* CSS */\n.my-class { color: red; }\n\n<!-- HTML -->\n<div class=\"my-class\">..."}]})

        # Missing doctype
        if i == 0 and s.lower() != "<!doctype html>":
            bugs.append({"id":"NO_DOCTYPE","name":"Missing <!DOCTYPE html>","severity":"MEDIUM",
                "line":1,"lineSnippet":s,
                "desc":"HTML documents should start with <!DOCTYPE html> to ensure standards mode.",
                "fixes":[{"label":"Fix — Add DOCTYPE","approach":"First line must be DOCTYPE declaration.",
                          "code":"<!DOCTYPE html>\n<html lang=\"en\">\n..."}]})

    return bugs


# ══════════════════════════════════════════════════════════════════════
# UNIVERSAL PATTERN RULES (all languages)
# ══════════════════════════════════════════════════════════════════════

UNIVERSAL_RULES = [
    {"id":"HARDCODED_SECRET","name":"Hardcoded Secret / Credential","severity":"CRITICAL",
     "langs":["python","java","javascript","c","cpp"],
     "pattern":re.compile(r'(password|secret|api_key|token|passwd|pwd|auth)\s*[=:]\s*["\'][^"\']{3,}["\']',re.IGNORECASE),
     "desc":"Credentials in source code exposed in version control and logs.",
     "fixes":[
         {"label":"Fix A — Environment variable","approach":"Read from environment.",
          "code":"# Python\npassword = os.environ.get('DB_PASSWORD')\n\n// Java\nString pwd = System.getenv(\"DB_PASSWORD\");\n\n// JS\nconst pwd = process.env.DB_PASSWORD;"},
         {"label":"Fix B — Config file","approach":"Store in config outside source control.",
          "code":"# .env file (add to .gitignore)\nDB_PASSWORD=mysecret"},
     ]},
    {"id":"TODO_FIXME","name":"TODO / FIXME Comment","severity":"LOW",
     "langs":["python","java","javascript","c","cpp"],
     "pattern":re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG)\b"),
     "desc":"Unresolved TODO/FIXME comments indicate incomplete or problematic code.",
     "fixes":[
         {"label":"Fix — Resolve or ticket","approach":"Either fix the issue or create a proper issue tracker ticket.",
          "code":"# Replace TODO with actual implementation\n# Or: raise NotImplementedError('see issue #123')"},
     ]},
    {"id":"MAGIC_NUMBER","name":"Magic Number — Use Named Constant","severity":"LOW",
     "langs":["python","java","javascript","c","cpp"],
     "pattern":re.compile(r"[^a-zA-Z_\"\'](?<!['\"])\b([2-9]\d{1,}|[1-9]\d{2,})\b(?!['\"])"),
     "desc":"Raw numbers in code are unclear. Name them as constants for readability.",
     "fixes":[
         {"label":"Fix — Named constant","approach":"Extract magic number to a named constant.",
          "code":"# Python\nMAX_RETRIES = 100\nfor _ in range(MAX_RETRIES): ...\n\n// Java\nfinal int MAX_RETRIES = 100;\n\n// JS\nconst MAX_RETRIES = 100;"},
     ]},
]


def check_universal(code, lang):
    bugs  = []
    lines = code.split("\n")
    for rule in UNIVERSAL_RULES:
        if lang not in rule.get("langs", []):
            continue
        if rule["pattern"].search(code):
            ln = 1
            for i,line in enumerate(lines):
                if rule["pattern"].search(line):
                    ln = i + 1
                    break
            bugs.append({"id":rule["id"],"name":rule["name"],"severity":rule["severity"],
                "line":ln,"lineSnippet":lines[ln-1].strip() if 0<ln<=len(lines) else "",
                "desc":rule["desc"],"fixes":rule["fixes"]})
    return bugs


# ══════════════════════════════════════════════════════════════════════
# PYTHON PATTERN RULES
# ══════════════════════════════════════════════════════════════════════

PYTHON_PATTERNS = [
    {"id":"OFF_BY_ONE","name":"Off-by-One Error","severity":"HIGH",
     "pattern":re.compile(r"range\s*\(\s*len\s*\(.+?\)\s*\+\s*1\s*\)"),
     "desc":"Loop iterates one past the valid range — IndexError on last iteration.",
     "fixes":[
         {"label":"Fix A — range(len(arr))","approach":"Remove +1.",
          "code":"for i in range(len(arr)):\n    total += arr[i]"},
         {"label":"Fix B — for-each","approach":"No index needed.",
          "code":"for item in arr:\n    total += item"},
         {"label":"Fix C — sum()","approach":"Built-in for totals.",
          "code":"return sum(arr)"},
         {"label":"Fix D — Recursive","approach":"Recursive sum with correct base case.",
          "code":"def sum_arr(arr, i=0):\n    if i == len(arr): return 0\n    return arr[i] + sum_arr(arr, i+1)"},
     ]},
    {"id":"NULL_DEREF","name":"Null / None Dereference","severity":"CRITICAL",
     "pattern":re.compile(r"while\s+\w+\.\w+|return\s+\w+\.\w+\s*$",re.MULTILINE),
     "desc":"Attribute access on potentially None raises AttributeError.",
     "fixes":[
         {"label":"Fix A — Guard","approach":"Check None first.",
          "code":"if obj is None: return None\nreturn obj.val"},
         {"label":"Fix B — raise","approach":"Raise descriptive error.",
          "code":"if obj is None: raise ValueError('Expected non-None')"},
         {"label":"Fix C — Recursive + guard","approach":"Tail-recursive traversal.",
          "code":"def get_last(head):\n    if head is None: return None\n    if head.next is None: return head.val\n    return get_last(head.next)"},
     ]},
    {"id":"SQL_INJECTION","name":"SQL Injection (CWE-89)","severity":"CRITICAL",
     "pattern":re.compile(r'f["\'].*?(SELECT|INSERT|UPDATE|DELETE|WHERE).*?\{',re.IGNORECASE),
     "desc":"User data interpolated into SQL — arbitrary query execution.",
     "fixes":[
         {"label":"Fix A — Parameterized","approach":"Use ? placeholder.",
          "code":'cursor.execute("SELECT * FROM users WHERE name=?", (username,))'},
         {"label":"Fix B — ORM","approach":"SQLAlchemy eliminates raw SQL.",
          "code":"session.query(User).filter(User.name==username).first()"},
     ]},
    {"id":"RESOURCE_LEAK","name":"Resource Leak","severity":"HIGH",
     "pattern":re.compile(r"^\s*\w+\s*=\s*open\s*\(",re.MULTILINE),
     "desc":"File opened without context manager — leaks if exception occurs.",
     "fixes":[
         {"label":"Fix A — with","approach":"Auto-closes.",
          "code":"with open(f, 'r') as fh:\n    data = fh.read()"},
         {"label":"Fix B — pathlib","approach":"One-liner.",
          "code":"from pathlib import Path\ndata = Path(f).read_text()"},
     ]},
    {"id":"EVAL_PY","name":"Dangerous eval()","severity":"CRITICAL",
     "pattern":re.compile(r"\beval\s*\(|\bexec\s*\("),
     "desc":"eval/exec execute arbitrary code — RCE if input from user.",
     "fixes":[
         {"label":"Fix A — ast.literal_eval","approach":"Safe for literals only.",
          "code":"import ast\ndata = ast.literal_eval(user_input)"},
         {"label":"Fix B — json.loads","approach":"For JSON data.",
          "code":"import json\ndata = json.loads(user_input)"},
     ]},
    {"id":"FIB_NO_MEMO","name":"Fibonacci Without Memoization O(2^n)","severity":"HIGH",
     "pattern":re.compile(r"def\s+fib\w*\s*\([^)]*\)\s*:(?![\s\S]*lru_cache)(?![\s\S]*memo)",re.MULTILINE),
     "desc":"Naive recursive Fibonacci is O(2^n). fib(40) = 2 billion calls.",
     "fixes":[
         {"label":"Fix A — @lru_cache","approach":"O(n) with one decorator.",
          "code":"from functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n<=1: return n\n    return fib(n-1)+fib(n-2)"},
         {"label":"Fix B — Bottom-up DP","approach":"O(n) time O(1) space.",
          "code":"def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a"},
         {"label":"Fix C — Matrix exp","approach":"O(log n).",
          "code":"def fib(n):  # O(log n)\n    def mul(A,B):\n        return [[A[0][0]*B[0][0]+A[0][1]*B[1][0],\n                  A[0][0]*B[0][1]+A[0][1]*B[1][1]],\n                 [A[1][0]*B[0][0]+A[1][1]*B[1][0],\n                  A[1][0]*B[0][1]+A[1][1]*B[1][1]]]\n    def pw(M,n):\n        if n==1: return M\n        h=pw(M,n//2); r=mul(h,h)\n        return r if n%2==0 else mul(M,r)\n    return pw([[1,1],[1,0]],n)[0][1] if n else 0"},
         {"label":"Fix D — Memo dict","approach":"Manual memoization.",
          "code":"memo={}\ndef fib(n):\n    if n in memo: return memo[n]\n    if n<=1: return n\n    memo[n]=fib(n-1)+fib(n-2)\n    return memo[n]"},
     ]},
    {"id":"RECURSION_NO_BASE","name":"Recursion Without Base Case","severity":"HIGH",
     "pattern":re.compile(r"def\s+(\w+)\s*\([\s\S]{0,300}\1\s*\(",re.MULTILINE),
     "desc":"Recursive function may lack base case — RecursionError (stack overflow).",
     "fixes":[
         {"label":"Fix A — Explicit base","approach":"Define termination first.",
          "code":"def factorial(n):\n    if n<=1: return 1  # base case\n    return n*factorial(n-1)"},
         {"label":"Fix B — Iterative","approach":"No stack risk.",
          "code":"def factorial(n):\n    r=1\n    for i in range(2,n+1): r*=i\n    return r"},
         {"label":"Fix C — @lru_cache","approach":"Memoized recursion.",
          "code":"from functools import lru_cache\n@lru_cache(maxsize=None)\ndef factorial(n):\n    if n<=1: return 1\n    return n*factorial(n-1)"},
         {"label":"Fix D — Reduce","approach":"Functional style.",
          "code":"from functools import reduce\nfactorial=lambda n: reduce(lambda a,b:a*b, range(1,n+1),1)"},
     ]},
    {"id":"NESTED_LOOP","name":"O(n^2) Nested Loop","severity":"LOW",
     "pattern":re.compile(r"for\s+\w+\s+in[\s\S]{0,100}\n\s+for\s+\w+\s+in"),
     "desc":"Nested loop is O(n^2). Consider set/dict-based O(n) approach.",
     "fixes":[
         {"label":"Fix A — Set O(n)","approach":"O(1) set lookup.",
          "code":"seen=set()\nduplicates=[x for x in arr if x in seen or seen.add(x)]"},
         {"label":"Fix B — Counter","approach":"O(n) frequency map.",
          "code":"from collections import Counter\ncounts=Counter(arr)"},
         {"label":"Fix C — Two pointers","approach":"O(n log n) sort + O(n) scan.",
          "code":"arr.sort()\nl,r=0,len(arr)-1\nwhile l<r:\n    if arr[l]+arr[r]==target: ...\n    elif arr[l]+arr[r]<target: l+=1\n    else: r-=1"},
         {"label":"Fix D — DP table","approach":"Overlapping subproblems.",
          "code":"dp=[[0]*(m+1) for _ in range(n+1)]\nfor i in range(1,n+1):\n    for j in range(1,m+1):\n        dp[i][j]=dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])"},
     ]},
]


def check_python_patterns(code):
    bugs  = []
    lines = code.split("\n")
    for rule in PYTHON_PATTERNS:
        if rule["pattern"].search(code):
            ln = 1
            for i,line in enumerate(lines):
                if rule["pattern"].search(line):
                    ln = i+1; break
            bugs.append({"id":rule["id"],"name":rule["name"],"severity":rule["severity"],
                "line":ln,"lineSnippet":lines[ln-1].strip() if 0<ln<=len(lines) else "",
                "desc":rule["desc"],"fixes":rule["fixes"]})
    return bugs


# ══════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTOR
# ══════════════════════════════════════════════════════════════════════

def detect_language(code, hint=""):
    hint = hint.lower()
    if hint in ("python","java","c","cpp","javascript","js","html"): 
        return hint.replace("js","javascript")
    if re.search(r"public\s+class|System\.out|import\s+java\.", code): return "java"
    if re.search(r"<html|<!DOCTYPE|<div|<p>|<head>", code, re.IGNORECASE): return "html"
    if re.search(r"console\.log|const |let |var |=>|require\(|module\.exports", code): return "javascript"
    if re.search(r"#include\s*<|printf\s*\(|scanf\s*\(|int\s+main\s*\(", code):
        return "cpp" if re.search(r"cout|cin|namespace\s+std|vector<|string\s+\w+\s*=", code) else "c"
    return "python"


# ══════════════════════════════════════════════════════════════════════
# MASTER ANALYZE
# ══════════════════════════════════════════════════════════════════════

def analyze_code(code, lang_hint=""):
    lang = detect_language(code, lang_hint)
    all_bugs = []

    if lang == "python":
        all_bugs += check_python_syntax(code)
        all_bugs += check_python_ast(code)
        all_bugs += check_python_patterns(code)
    elif lang == "java":
        all_bugs += check_java(code)
    elif lang in ("c","cpp"):
        all_bugs += check_c_cpp(code, lang)
    elif lang == "javascript":
        all_bugs += check_javascript(code)
    elif lang == "html":
        all_bugs += check_html(code)

    all_bugs += check_universal(code, lang)

    # Deduplicate
    seen, final = set(), []
    for b in all_bugs:
        k = (b["id"], b["line"])
        if k not in seen:
            seen.add(k); final.append(b)

    SEV = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
    final.sort(key=lambda b: (SEV.get(b["severity"],4), b["line"]))

    n    = len(final)
    prob = min(0.45+n*0.13, 0.96) if n else 0.04
    sev  = min(0.30+n*0.16, 0.95) if n else 0.02

    return {"is_vulnerable":n>0,"bug_probability":round(prob,4),
            "severity":round(sev,4),"confidence":round(prob if n else 0.96,4),
            "issues":final,"line_count":len(code.split("\n")),
            "lang":lang,"mode":"multi-lang-heuristic"}


# ══════════════════════════════════════════════════════════════════════
# AI SUGGEST — uses Claude API to identify algorithm and give all
# possible implementations across complexity levels
# ══════════════════════════════════════════════════════════════════════

def call_claude_suggest(code, lang):
    """Call Claude API to identify what the code does and suggest all implementations."""
    try:
        import urllib.request, json as jsonlib
        prompt = f"""You are an expert software engineer and algorithm tutor.

The user has pasted the following {lang} code:

```{lang}
{code}
```

Your job:
1. Identify what algorithm or problem this code is trying to solve (e.g. "Fibonacci", "Factorial", "Binary Search", "Sorting", etc.)
2. Give the corrected version of the code
3. Then show ALL possible implementations of this algorithm from simplest to most optimal, covering:
   - Brute Force
   - Iterative
   - Recursive  
   - Dynamic Programming (if applicable)
   - Backtracking (if applicable)
   - Divide and Conquer (if applicable)
   - Optimal / Most Efficient

For each implementation include:
- Method name
- Time complexity
- Space complexity
- Clean working code in {lang}
- One sentence about when to use it

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "algorithm": "Name of the algorithm/problem",
  "description": "What this code is trying to do in one sentence",
  "corrected_code": "the fixed version of the original code",
  "implementations": [
    {{
      "method": "Brute Force",
      "time": "O(n^2)",
      "space": "O(1)",
      "when": "When input is small and simplicity matters",
      "code": "...working code..."
    }}
  ]
}}"""

        payload = jsonlib.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{"role":"user","content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type":"application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = jsonlib.loads(resp.read().decode("utf-8"))
            text = data["content"][0]["text"]
            return jsonlib.loads(text)

    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════
# LOAD TRAINED MODEL
# ══════════════════════════════════════════════════════════════════════

model  = None
DEVICE = None
try:
    sys.path.insert(0, "/Users/pawan/Downloads/hgsn_v3")
    from train import HGSN, predict as model_predict, load_model, M2_CONFIG
    import torch
    DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    CHECKPOINT = "/Users/pawan/Downloads/hgsn_v3/checkpoints/hgsn_best.pt"
    model      = load_model(CHECKPOINT)
except Exception as e:
    print(f"Info: Model not loaded ({e}). Heuristic engine active.")


# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model_loaded":model is not None,
                    "device":str(DEVICE) if DEVICE else "cpu"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or "code" not in data:
        return jsonify({"error":"Missing 'code' field"}), 400
    code = data["code"].strip()
    if not code: return jsonify({"error":"Empty code"}), 400

    lang   = data.get("lang","")
    result = analyze_code(code, lang)

    if model is not None:
        try:
            mp = model_predict(model, code, DEVICE)
            result["model_prediction"] = mp
            result["bug_probability"]  = round(max(result["bug_probability"],
                                                     mp.get("bug_probability",0)),4)
            result["mode"] = "hgsn-model"
        except Exception as ex:
            print(f"Model predict failed: {ex}")

    return jsonify(result)

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json(silent=True)
    if not data or "code" not in data:
        return jsonify({"error":"Missing 'code' field"}), 400
    code = data["code"].strip()
    if not code: return jsonify({"error":"Empty code"}), 400
    lang = data.get("lang") or detect_language(code)
    result = call_claude_suggest(code, lang)
    return jsonify(result)

if __name__ == "__main__":
    print("\n BugLens API at http://localhost:5001")
    print("   Open index.html to use the UI\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
