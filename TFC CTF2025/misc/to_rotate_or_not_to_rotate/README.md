# TO ROTATE, OR NOT TO ROTATE （TFCCTF2025|MiSC）

> One fine evening, A had an important mission: pass a bunch of critical configurations to his good friend B.
>
> These configs were patterns—very serious, technical things—based on segments over a neat little 3×3 grid.
>
> But there was a problem. B was wasted.
> Like, “talking to the couch and thinking it’s a microwave” level drunk. So when A carefully handed over each configuration, B took one look at it, said, “Whoa, cool spinny lines!”—and rotated it randomly. Then, to add insult to intoxication, he shuffled the order of all the patterns. Absolute chaos.
>
> Now A has a challenge: figure out which drunkenly-distorted pattern maps back to which original configuration. If he gets it all right, B promises (in slurred speech) to give him something very important: the flag.

Files: to_rotate,or_not_to_rotate.zip

*My English is very poor, and most of the content was translated by AI. The original post is at: [TFC CTF2025\]Misc-to_rotate_or_not_to_rotate-CSDN博客](https://blog.csdn.net/astmling_/article/details/151051225)*

## Introduction

The container runtime for this competition was way too short—just 5 minutes :(
(It was later extended to 15 minutes, though.)

This means we need to analyze the code provided in the problem first, then start the environment to solve it.

```python
import os, sys, random

POINTS = [(x, y) for x in range(3) for y in range(3)]

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def valid_segment(a, b):
    if a == b: return False
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return gcd(dx, dy) == 1 and 0 <= a[0] <= 2 and 0 <= a[1] <= 2 and 0 <= b[0] <= 2 and 0 <= b[1] <= 2

SEGMENTS = []
for i in range(len(POINTS)):
    for j in range(i+1, len(POINTS)):
        a, b = POINTS[i], POINTS[j]
        if valid_segment(a, b):
            A, B = sorted([a, b])
            SEGMENTS.append((A, B))
assert len(SEGMENTS) == 28
SEG_INDEX = {SEGMENTS[i]: i for i in range(28)}

def rot_point(p, k):
    x, y = p; cx, cy = 1, 1
    x0, y0 = x - cx, y - cy
    for _ in range(k % 4):
        x0, y0 = -y0, x0
    return (x0 + cx, y0 + cy)

def rot_segment(seg, k):
    a, b = seg
    ra, rb = rot_point(a, k), rot_point(b, k)
    A, B = sorted([ra, rb])
    return (A, B)

def canon_bits(segs):
    vals = []
    for k in range(4):
        bits = 0
        for (a, b) in segs:
            A, B = sorted([a, b])
            rs = rot_segment((A, B), k)
            bits |= (1 << SEG_INDEX[rs])
        vals.append(bits)
    return min(vals)

def parse_pattern(m, lines):
    segs, seen = [], set()
    if m <= 0: raise ValueError("m must be > 0")
    for ln in lines:
        x1, y1, x2, y2 = map(int, ln.split())
        a, b = (x1, y1), (x2, y2)
        if not valid_segment(a, b):
            raise ValueError("invalid segment")
        A, B = sorted([a, b])
        if (A, B) in seen: raise ValueError("duplicate segment")
        seen.add((A, B))
        segs.append((A, B))
    return segs

def mutate_pattern(segs):
    k = random.randrange(4)
    out = []
    for (a, b) in segs:
        ra, rb = rot_point(a, k), rot_point(b, k)
        if random.getrandbits(1): ra, rb = rb, ra
        out.append((ra, rb))
    random.shuffle(out)
    return out

def println(*xs):
    sys.stdout.write(" ".join(str(x) for x in xs) + "\n")
    sys.stdout.flush()

def read_int():
    s = sys.stdin.readline()
    if not s: raise EOFError
    return int(s.strip())

def read_line():
    s = sys.stdin.readline()
    if not s: raise EOFError
    return s.strip()

# --------- main protocol ----------
def main():
    random.seed(os.urandom(16))
    Q     = 120         
    MAXN  = 40000000
    FLAG  = "TFCCTF{fake_flag}"

    println("=== To rotate, or not to rotate ===")
    sys.stdout.flush()

    # Phase 1
    Ns = [random.randint(1, MAXN) for _ in range(Q)]
    canon2N = {}
    stored = []  
    for i, N in enumerate(Ns, 1):
        println(f"N_{i}:", N)
        try:
            m = read_int()
            lines = [read_line() for _ in range(m)]
            segs = parse_pattern(m, lines)
            c = canon_bits(segs)
            if c in canon2N and canon2N[c] != N:
                println("Error: duplicate pattern modulo rotation used for a different N. Bye.")
                return
            canon2N[c] = N
            stored.append((N, segs, c))
            println("OK")
        except Exception as e:
            println("Input error:", e)
            return

    println("=== Phase 2 ===")
    order = list(range(Q))
    random.shuffle(order)
    correct = 0
    for idx in order:
        N, segs, c = stored[idx]
        mut = mutate_pattern(segs)
        println("MutatedPattern:")
        println(len(mut))
        for (a, b) in mut:
            println(a[0], a[1], b[0], b[1])
        println("Your answer for N?")
        try:
            ans = read_int()
        except Exception:
            println("Bad answer. Bye.")
            return
        expected = canon2N.get(canon_bits(mut), None)
        if expected is None:
            println("Internal error: unknown pattern. Bye."); return
        if ans == expected:
            println("OK")
            correct += 1
        else:
            println("Wrong (expected", expected, ")")

    if correct == Q:
        println("All correct! Here is your flag:")
        println(FLAG)
    else:
        println(f"You solved {correct}/{Q}. No flag.")

if __name__ == "__main__":
    main()
```

We need to complete Phase 1 and Phase 2 to get the flag.

------

## Analyzing Phase 1.

In a 3x3 grid, draw m segments. These m segments are respectively rotated by 0°, 90°, 180°, and 270° around the center point (1,1).

Other segments must not overlap with any previous segments or their rotated versions.

1. A pattern consisting solely of the segment from (0,0) to (0,1) cannot coexist with a pattern consisting solely of the segment from (0,2) to (1,2) (because these two patterns would coincide after rotation, violating the given condition). The red segments represent the original ones, and the blue segments represent the rotated versions.

（Note: The example illustrates that if two segments are rotational symmetries of each other around (1,1), they are considered duplicates and cannot both be included. The rule requires that no segment (or its rotated version) overlaps with any other segment in the set.)

![img](https://i-blog.csdnimg.cn/direct/4ad7bd437e4e4e9ead6ed37c35dfb316.png)

 2.The pattern composed of the three segments:

- (0,0) to (1,0)
- (1,0) to (1,1)
- (1,1) to (0,1)

After being rotated by 0°, 90°, 180°, and 270° around the center (1,1), it may only partially overlap with other rotated versions of itself, but it is still acceptable (i.e., it passes the condition).

This means that even if the rotated versions do not fully coincide (only partial overlapping occurs), the pattern is allowed. The rule prohibits exact overlaps or conflicts that would make segments indistinguishable, but partial overlaps (as long as they do not entirely cover or duplicate existing segments) are permitted.

In other words:

- The condition forbids segments that are identical to any rotated version (0°/90°/180°/270°) of an existing segment.
- However, if a segment and its rotations only partially overlap with others (without being exact duplicates), it is considered valid.

This example shows that the rule is about avoiding identical copies (due to rotation) rather than preventing all overlaps. Partial overlaps are acceptable.

![img](https://i-blog.csdnimg.cn/direct/c254b894edec404a94a32a7e69586789.png)

3.Patterns composed of different segments (e.g., a pattern with one segment and a pattern with three segments) do not exist on the same layer and do not interfere with each other.
(For example, a pattern made of a single segment and a pattern made of three segments are independent and do not affect each other.)

This means:

- Each distinct set of segments (each "pattern") is considered independently.
- The rotation condition applies per pattern: a pattern and its rotated versions (0°, 90°, 180°, 270°) must not exactly duplicate another pattern (in terms of segment positions).
- However, patterns with different numbers of segments (e.g., a one-segment pattern and a three-segment pattern) are compared separately. They are not required to be unique relative to each other. Even if they partially overlap after rotation, it is allowed as long as no two patterns are identical.

In short:

- Uniqueness is enforced only within patterns of the same "type" (same number of segments and same configuration). Patterns with different structures are exempt from conflicting with each other.

  

------

Next, you need to input 1000 patterns that meet the above conditions to proceed to Phase 2.

***Note:***
*The example earlier (with 120 patterns) might be a simplified requirement, but the actual server expects 1000 patterns. Ensure your solution generates enough valid patterns to meet Q=1000.*

### Analyzing Phase 2.

```
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
def rot_point(p, k):
    x, y = p; cx, cy = 1, 1
    x0, y0 = x - cx, y - cy
    for _ in range(k % 4):
        x0, y0 = -y0, x0
    return (x0 + cx, y0 + cy)
def rot_segment(seg, k):
    a, b = seg
    ra, rb = rot_point(a, k), rot_point(b, k)
    A, B = sorted([ra, rb])
    return (A, B)
```

The server randomly rotates patterns (0°, 90°, 180°, or 270°) and may reverse segment directions (e.g., swapping point order). 

### Solving !

In Phase1,

duplicate unit segments were found, so we need to extract valid segments.

```
dp = ["1 0 0 0", "1 0 0 1", "0 0 1 1", "1 1 0 1", "1 1 1 0", "0 0 0 1", "1 1 2 2", "0 0 1 2", "0 1 2 0", "0 1 0 2"]
```



Then, map the segments to their corresponding pattern index **N** and store them in a dictionary.



In Phase2,

Simply tracking raw point-to-point segments would be inefficient due to the high processing overhead. 

Instead, we need a normalized representation of segments to handle these transformations consistently.

So we convert these point-to-point connections into standardized unit segments.

```
segment_map = {'0 0 1 0': 0, '1 0 2 0': 1, '0 1 1 1': 2, '1 1 2 1': 3, '0 2 1 2': 4, '1 2 2 2': 5, '0 0 0 1': 6, '0 1 0 2': 7, '1 0 1 1': 8, '1 1 1 2': 9, '2 0 2 1': 10, '2 1 2 2': 11, '0 0 1 1': 12, '1 1 2 2': 13, '0 1 1 0': 14, '1 2 2 1': 15, '0 1 1 2': 16, '1 0 2 1': 17, '0 2 1 1': 18, '1 1 2 0': 19, '0 0 1 2': 20, '0 0 2 1': 21, '0 1 2 2': 22, '0 1 2 0': 23, '0 2 1 0': 24, '1 0 2 2': 25, '0 2 2 1': 26, '1 2 2 0': 27, '1 0 0 0': 0, '2 0 1 0': 1, '1 1 0 1': 2, '2 1 1 1': 3, '1 2 0 2': 4, '2 2 1 2': 5, '0 1 0 0': 6, '0 2 0 1': 7, '1 1 1 0': 8, '1 2 1 1': 9, '2 1 2 0': 10, '2 2 2 1': 11, '1 1 0 0': 12, '2 2 1 1': 13, '1 0 0 1': 14, '2 1 1 2': 15, '1 2 0 1': 16, '2 1 1 0': 17, '1 1 0 2': 18, '2 0 1 1': 19, '1 2 0 0': 20, '2 1 0 0': 21, '2 2 0 1': 22, '2 0 0 1': 23, '1 0 0 2': 24, '2 2 1 0': 25, '2 1 0 2': 26, '2 0 1 2': 27}
```

Then, we can rotate each pattern by the four angles (0°, 90°, 180°, and 270°).

Indeed, each pattern is unique, and we can try to use a mask to represent it.

> b'N_91: 29675970\r\n'
> b'3'
> b'1 0 0 1'
> b'1 1 0 1'
> b'1 1 1 0'
> ['0 1 1 0', '0 1 1 1', '1 0 1 1'] //0
> bitmask: 16644 
> ['1 0 2 1', '1 0 1 1', '1 1 2 1']//90
> bitmask: 131336 
> ['1 2 2 1', '1 1 2 1', '1 1 1 2']//180
> bitmask: 33288 
> ['0 1 1 2', '1 1 1 2', '0 1 1 1']//270

Then, the script can extract the corresponding value **N** (the pattern index) from the dictionary based on the bitmask and submit it. After completing this process **1000 times** (i.e., correctly matching 1000 obfuscated patterns), the challenge will be solved.

### Final exp

```
from pwn import *
from itertools import combinations
import os
import re


def rot_point(p, k):
    x, y = p;
    cx, cy = 1, 1
    x0, y0 = x - cx, y - cy
    for _ in range(k % 4):
        x0, y0 = -y0, x0
    return (x0 + cx, y0 + cy)


def segments_to_strings(segments):
    """
    Args:
        segments:  [((x1,y1), (x2,y2)), ...]
    Returns:
         ["x1 y1 x2 y2", ...]
    """
    result = []
    for seg in segments:
        a, b = seg
        sorted_seg = sorted([a, b], key=lambda point: (point[0], point[1]))
        a, b = sorted_seg
        result.append(f"{a[0]} {a[1]} {b[0]} {b[1]}")
    return result


def mutate_pattern(segs, i):
    k = i
    out = []
    for (a, b) in segs:
        ra, rb = rot_point(a, k), rot_point(b, k)
        if random.getrandbits(1): ra, rb = rb, ra
        out.append((ra, rb))

    return segments_to_strings(out)


def segments_to_bitmask(segment_strings):
    segment_map = {'0 0 1 0': 0, '1 0 2 0': 1, '0 1 1 1': 2, '1 1 2 1': 3, '0 2 1 2': 4, '1 2 2 2': 5, '0 0 0 1': 6,
                   '0 1 0 2': 7, '1 0 1 1': 8, '1 1 1 2': 9, '2 0 2 1': 10, '2 1 2 2': 11, '0 0 1 1': 12, '1 1 2 2': 13,
                   '0 1 1 0': 14, '1 2 2 1': 15, '0 1 1 2': 16, '1 0 2 1': 17, '0 2 1 1': 18, '1 1 2 0': 19,
                   '0 0 1 2': 20, '0 0 2 1': 21, '0 1 2 2': 22, '0 1 2 0': 23, '0 2 1 0': 24, '1 0 2 2': 25,
                   '0 2 2 1': 26, '1 2 2 0': 27, '1 0 0 0': 0, '2 0 1 0': 1, '1 1 0 1': 2, '2 1 1 1': 3, '1 2 0 2': 4,
                   '2 2 1 2': 5, '0 1 0 0': 6, '0 2 0 1': 7, '1 1 1 0': 8, '1 2 1 1': 9, '2 1 2 0': 10, '2 2 2 1': 11,
                   '1 1 0 0': 12, '2 2 1 1': 13, '1 0 0 1': 14, '2 1 1 2': 15, '1 2 0 1': 16, '2 1 1 0': 17,
                   '1 1 0 2': 18, '2 0 1 1': 19, '1 2 0 0': 20, '2 1 0 0': 21, '2 2 0 1': 22, '2 0 0 1': 23,
                   '1 0 0 2': 24, '2 2 1 0': 25, '2 1 0 2': 26, '2 0 1 2': 27}
    bitmask = 0
    for seg_str in segment_strings:
        if seg_str in segment_map:
            index = segment_map[seg_str]
            bitmask |= (1 << index)
        else:
            print(f"error:  '{seg_str}'")

    return bitmask

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
dp = ["1 0 0 0", "1 0 0 1", "0 0 1 1", "1 1 0 1", "1 1 1 0", "0 0 0 1", "1 1 2 2", "0 0 1 2", "0 1 2 0", "0 1 0 2"]
# conn=remote("to-rotate-4c6ef0a8d44330dc.challs.tfcctf.com",1337, ssl=True)
conn = process(["python", r"server.py"])
print(conn.recvuntil(b"N_"))
t1 = conn.recvline()
print(t1)
N = re.findall(": (.*)\n", t1.decode())[0]


def get_all_combinations(lst, n):
    return list(combinations(lst, n))


t = 0
cc = 0
found = False
dict = {}
for n in range(1, 9):
    result = get_all_combinations(dp, n)
    for i, comb in enumerate(result, 1):
        if str(comb) == "('0 0 1 2', '0 1 0 2')" or str(comb) == "('1 1 0 1', '0 0 1 2', '0 1 0 2')" or str(
                comb) == "('1 1 1 0',)" or str(comb) == "('1 1 2 2',)" or str(comb) == "('0 1 2 0',)" or str(
                comb) == "('1 1 1 0', '0 1 2 0')" or str(comb) == "('0 1 0 2',)" or str(
                comb) == "('1 1 0 1', '0 1 0 2')" or str(comb) == "('0 1 2 0', '0 1 0 2')" or str(
                comb) == "('1 1 0 1', '0 1 2 0', '0 1 0 2')": //Exclude invalid patterns
            continue
        tt = []
        print(str(n).encode())
        cc += 1
        conn.sendline(str(n).encode())
        dp1 = []
        for j in comb:
            dp1.append([(list(map(lambda x: int(x), str(j).split()))[:2])] + [
                (list(map(lambda x: int(x), str(j).split()))[2:])])
            conn.sendline(str(j).encode())
            print(str(j).encode())
        for rotation in range(4):
            dp2 = mutate_pattern(dp1, rotation)
            print(dp2)
            bitmask = segments_to_bitmask(dp2)
            print(f"bitmask: {bitmask} ")
            tt.append(bitmask)
        for x in tt:
            dict[x] = N
        print(conn.recvline())
        text = (conn.recvline())
        print(text)
        if b"=== Phase 2 ===" in text:
            found = True
            break
        N = (re.findall(": (.*)\n", text.decode()))[0]

    if found == True:
        break
    t += len(result)

conn.recvline()
for i in range(1000):
    t = (conn.recvline().strip().decode())
    if "TFCCTF" in t:
        print(t)
        break
    else:
        t = int(t)
        dppp = []
        for i in range(t):
            dppp.append(conn.recvline().strip().decode())
    bitmask = segments_to_bitmask(dppp)
    print(dppp)
    print(f"bitmask: {bitmask}")
    text = (dict[bitmask])
    print(conn.recvline())
    conn.sendline(text.encode())
    print(conn.recvline())
    print(conn.recvline())
print(conn.recvline())
```

*Running the script will return the flag.*

```
TFCCTF{\xe5\x9b\x9e\xe8\xbb\xa2_\xe9\xbb\x84\xe9\x87\x91\xe9\x95\xb7\xe6\x96\xb9\xe5\xbd\xa2_\xe6\xaf\x94\xe7\x8e\x87}
```

### Flag!!!

```
TFCCTF{回転_黄金長方形_比率}
```
