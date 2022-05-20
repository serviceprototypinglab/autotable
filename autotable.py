import pandas as pd
import datetime
import random
import math

fn = "vbv_M0114.csv"
autosep = ";" # FIXME detect automagically

def printcolor(s, col):
    color = {}
    color["red"] = "\033[0;31m"
    color["yellow"] = "\033[1;33m"
    color["blue"] = "\033[0;34m"
    reset = "\033[0m"
    print(f"{color[col]}{s}{reset}")

def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

class SeriesKnowledge:
    def __init__(self):
        self.values = set()
        self.valueprob = {}
        self.incrementing = None
        self.incrementing_orig = None
        self.incrementing_base = None
        self.incrementing_step = None
        self.min = None
        self.max = None
        self.groupedvalues = {}
        self.istime = None

    def dump(self):
        prob = "(too many values)"
        if len(self.valueprob) < 5:
            prob = self.valueprob
        return f"[{len(self.values)} values, incrementing: {self.incrementing}:{self.incrementing_base}:{self.incrementing_step}, min/max: {self.min}/{self.max}, probs: {prob}...]"

class Generator:
    def __init__(self, columns, separator, colmap):
        self.columns = columns
        self.sep = separator
        self.colmap = colmap

    def header(self):
        return self.sep.join(self.columns)

    def generate(self):
        data = []
        ndata = {}

        # FIXME hardcoded hack
        mapping = {"Netto": ("Brutto", -random.randrange(3, 5) / 10)}

        for col in colmap:
            if col in mapping:
                ocol, derivation = mapping[col]
                val = round(float(ndata[ocol]) + derivation, 1)
                data.append(str(val))
                ndata[ocol] = val
                continue

            sk = colmap[col]
            val = "*"
            if sk.incrementing:
                diff = sk.incrementing_step
                if diff not in (-1, 0, 1):
                    diff = random.randrange(abs(round(diff)) * 2) * sgn(diff)
                    if type(sk.incrementing_base) == float:
                        diff += random.random()
                sk.incrementing_base += diff
                val = sk.incrementing_base
            elif len(sk.values) == 1:
                val = list(sk.values)[0]
            elif sk.min and sk.max:
                minval = sk.min
                maxval = sk.max
                choices = []

                usegrouping = False

                # FIXME hardcoded groupby column
                if "Klasse" in ndata and ndata["Klasse"] in sk.groupedvalues and usegrouping:
                    groupedvalues = sk.groupedvalues[ndata["Klasse"]]
                    minval = groupedvalues[0]
                    maxval = groupedvalues[-1]
                    useprob = True
                    for v in groupedvalues:
                        choices.append(v)
                else:
                    useprob = True
                    for v in sk.valueprob:
                        for i in range(sk.valueprob[v]):
                            choices.append(v)

                useprob = False

                fac = 1
                val = random.randrange(int(minval * 10 ** fac), int(maxval * 10 ** fac)) / 10 ** fac

                if useprob:
                    validx = random.randrange(len(choices))
                    window = int(len(choices) / 10)
                    validxmin = validx - window
                    validxmax = validx + window
                    if validxmin < 0:
                        validxmin = 0
                    if validxmax >= len(choices):
                        validxmax = len(choices) - 1
                    val = choices[random.randrange(validxmin, validxmax + 1)]
            else:
                choices = []
                for v in sk.valueprob:
                    for i in range(sk.valueprob[v]):
                        choices.append(v)
                val = random.choice(choices)

            if sk.istime:
                val = datetime.datetime.fromtimestamp(val).strftime("%d.%m.%Y %H:%M:%S.%f")
                val = val[:-3]

            if val is None:
                val = ""
            data.append(str(val))
            ndata[col] = val

        return self.sep.join(data)

df = pd.read_csv(fn, sep=autosep)
df = df[::-1]

colmap = {}
for col in df.columns:
    colmap[col] = SeriesKnowledge()

itercar = 0
for row in df.iterrows():
    itercar += 1
    rowdata = row[1]
    # FIXME: hardcoded column, needs autodetection
    rtime = rowdata[list(df.columns).index("Empfangszeit")]
    rtimestamp = datetime.datetime.strptime(rtime, "%d.%m.%Y %H:%M:%S.%f")

    printcolor(f"@ {rtime}", "blue")
    data = str(rowdata.values).replace("\n", "")
    printcolor(f"→ data {data}", "yellow")

    for val, col in zip(rowdata.values, colmap):
        if type(val) == float and math.isnan(val):
            val = None
        print(",", val, "@", col, type(val))

        sk = colmap[col]
        sk.values.add(val)
        sk.valueprob[val] = sk.valueprob.get(val, 0) + 1

        # FIXME hardcoding again
        if col == "Empfangszeit":
            stamp = rtimestamp.timestamp()
            val = stamp
            sk.istime = True

        if type(val) == int or type(val) == float:
            if sk.incrementing_base is None:
                sk.incrementing_base = val
                sk.incrementing_orig = val
            else:
                diff = val - sk.incrementing_base
                if sk.incrementing_step is None:
                    sk.incrementing_step = diff
                if diff not in (1, 0, -1):
                    sk.incrementing_step = (val - sk.incrementing_orig) / len(sk.values)
                if sgn(diff) == sgn(sk.incrementing_step):
                    if sk.incrementing is None:
                        sk.incrementing = True
                    sk.incrementing_base = val
                else:
                    sk.incrementing = False

        if type(val) in (int, float):
            if sk.min is None or val < sk.min:
                sk.min = val
            if sk.max is None or val > sk.max:
                sk.max = val

            # FIXME: hardcoded groupby column
            groupval = rowdata[list(df.columns).index("Klasse")]
            sk.groupedvalues[groupval] = sk.groupedvalues.get(groupval, []) + [val]

for col in colmap:
    sk = colmap[col]
    for groupval in sk.groupedvalues:
        sk.groupedvalues[groupval].sort()

print("---")
for col in colmap:
    print("-- Distinct values:", colmap[col].dump(), "@", col)

g = Generator(df.columns, autosep, colmap)
print(g.header())
for i in range(10):
    print(g.generate())

print("Generating ten million entries...")
f = open("gen.csv", "w")
print(g.header(), file=f)
for i in range(10 * 1000000):
    print(g.generate(), file=f)
f.close()

df = pd.read_csv("gen.csv", sep=";")
print(df)
