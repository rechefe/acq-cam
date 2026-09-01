# Finding GPS satellites with a memory chip instead of a calculator

*This explains what our design does and how it differs from the standard approach.
No signal-processing background needed — every technical term is defined the first
time it appears. For the version written for specialists, see
[`gps_findings.md`](gps_findings.md).*

## 1. The problem: a very hard game of "who's out there?"

A GPS receiver's first job, before it can tell you where you are, is simply to work
out **which satellites it can currently hear**. This step is called *acquisition*,
and it is the part that burns the most chip area and battery.

Here is the situation, by analogy.

Imagine 32 drummers, each playing their own distinctive 1023-beat pattern, over and
over, one full pattern per millisecond. Every drummer has a *different* pattern, and
the patterns are designed so that no two sound alike. They all play at once, on top
of each other. You are standing in a field with a microphone, and you want to know
which drummers are within earshot.

Three things make this hard:

**The noise is overwhelming.** The drummers are so far away that the background hiss
is roughly **300 times more powerful** than all the drumming combined. If you listen
to any single instant, you hear essentially pure noise. The signal is not faint — it
is *invisible* at any single moment.

**You don't know where in the pattern anyone is.** The pattern repeats every
millisecond, and you have no idea whether a given drummer is on beat 1 or beat 700
right now. That is 1023 possibilities per drummer. (We actually check twice per beat,
so 2046.) This unknown is called the **code phase**.

**You don't know how fast anyone is moving.** GPS satellites move fast enough that
their signal arrives slightly stretched or compressed — the same effect that makes an
ambulance siren drop in pitch as it passes. So each drummer's pattern arrives at a
slightly wrong speed, and you don't know which. This unknown is called the **Doppler
shift**, and we have to consider about 41 possible speeds.

Multiply it out:

```
32 drummers  x  1023 positions  x  41 speeds  =  1.3 million combinations
```

Every one of those 1.3 million combinations has to be checked, against a signal
buried 300x below the noise. That is the whole problem.

## 2. The one idea that makes any of this possible

If the signal is invisible at every single instant, how can it ever be found?

**Because you don't look at one instant. You look at thousands and add up the
evidence.**

Think of it as a biased coin. Take a single measurement and compare it against what a
particular drummer's pattern predicts. Because of the noise, that comparison is right
only about **53% of the time** — barely better than a coin flip. Useless on its own.

But do it 4092 times and count:

- Checking the **wrong** drummer, or the right drummer at the **wrong** position:
  each comparison is a fair coin. You expect about **2046 matches**.
- The **right** drummer at the **right** position: each comparison is a 53% coin.
  You expect about **2137 matches**.

That gap — about 90 out of 4092, roughly 2% — is small, but it is *consistent*, and
random luck almost never produces it. That is the entire basis of GPS. One coin tells
you nothing; four thousand coins tell you everything.

Everything that follows is just different ways of counting those coins.

## 3. The standard approach: compute a score

The conventional receiver treats this as arithmetic. For each drummer and each
possible speed, it runs a calculation that scores **all 1023 positions at once**,
using a well-known mathematical shortcut called the Fast Fourier Transform. Then it
takes the highest score and asks whether it is high enough to be real.

It works well and it is the industry standard. Its cost is arithmetic:

> **About 32 million multiply-and-add operations per millisecond of signal**, for one
> pass over all satellites and speeds.

Multipliers are the expensive part of a chip — they take area and they burn power. An
acquisition engine built this way is essentially a large, hungry calculator.

## 4. Our approach: let the memory do the comparing

We replace the calculator with a special kind of memory called a **CAM**
(content-addressable memory).

Ordinary memory works like a numbered locker room: you give it an address, it hands
back what is inside. A CAM works backwards. You hand it *content*, and it tells you
which lockers hold something similar — **checking every locker simultaneously, in a
single clock tick**. Ours is an *approximate-match* CAM, meaning it does not demand a
perfect match; it answers "close enough or not?" for every row at once.

So we store the drummers' patterns in the CAM's rows, feed in what the microphone
heard, and the chip answers, in one tick:

> "Rows 7, 12 and 30 are close enough. The rest are not."

**The trick that makes this legitimate:** comparing two strings of bits and counting
how many differ is *mathematically the same operation* as the multiply-and-add that
the standard method performs. But the CAM does the counting inside the memory itself,
as a physical side-effect of reading it. There are no multipliers anywhere.

### Where each unknown goes

We have unknowns to search and two places to put each one: **stored as rows** (costs
chip area) or **swept over time** (costs clock cycles). The rule we arrived at:

> **Sweep the unknowns that are cheap to undo. Store the ones that are not.**

| Unknown | Where it goes | Why |
|---|---|---|
| Which drummer (32) | Stored as rows | All 32 checked every tick — this is the free parallelism |
| Speed / Doppler (41) | Stored as rows | Undoing a speed error takes real arithmetic; pre-storing it is free |
| Position (2046) | Swept, one per tick | Just slide the listening window forward by one sample |
| Wave phase (4) | Swept, and **free** | See below |

That fourth unknown deserves a note. Besides not knowing *where* in the pattern a
drummer is, we also do not know whether the incoming wave happens to be at a peak or
a trough at the moment we start listening. It is a nuisance — we do not want to
measure it, we just have to not be defeated by it. It turns out that shifting it by a
quarter-turn amounts to **swapping two wires and flipping some signs**. No arithmetic
at all. So we try all four quarter-turns and accept whichever one fires. That costs
four clock ticks and zero chip area.

### The result

The whole search becomes:

```
for each of 41 speeds:              <- stored in rows, all checked in parallel
    for each of 2046 positions:     <- one clock tick each
        for each of 4 wave phases:  <- free: wire swaps
            ask the CAM: which rows are close enough?
```

## 5. Side by side

| | Standard | Ours |
|---|---|---|
| Core operation | Multiply and add | Compare and count |
| Hardware | Multipliers, adders, working memory | One memory array |
| Cost per pass | ~32 million multiply-adds | 0.65-5.4 million bits of stored patterns |
| Satellites per step | One at a time | All 32 at once |
| Output | A numeric score per combination | Yes / no per combination |
| Time to find a weak satellite | 6 ms | 17 ms |

That last row is the honest headline, and the next section unpacks it.

## 6. What we measured

We simulated the whole thing and compared against the standard method at a
realistically weak signal level.

**Time needed to reliably find a satellite** — 90% success, with false alarms held
to 1% across all 1.3 million combinations:

| Method | Time |
|---|---|
| Standard, full precision | 6 ms |
| Standard, but reading the same coarse input we use | 11 ms |
| **Ours (CAM)** | **17 ms** |

The middle row matters. Real GPS chips do not record the signal precisely — they
record a very coarse version of it, four levels rather than an exact number, because
that is vastly cheaper and it barely hurts. That coarseness costs roughly a factor of
two in time, and **every** GPS receiver already pays it. It is not our design's
fault, so charging it to us would be misleading.

The fair comparison is therefore 11 ms against 17 ms. **Our approach costs about 50%
more time than an equivalent conventional design — and in exchange it uses no
multipliers at all and checks all 32 satellites simultaneously.**

For context on whether that is good: an earlier study in this same project applied
the same memory chip to Bluetooth, where it came out roughly *ten times* worse than
the conventional approach. The difference is that Bluetooth needed a precise *number*
out of the chip, and a yes/no memory cannot give you one. GPS acquisition only needs
to know *which* satellite and *roughly* where — and "roughly" is exactly what a
yes/no answer provides. The task fits the tool.

## 7. The dial: chip size against time

There is one design knob, and it is the most useful result we have.

Instead of one long comparison across the whole pattern, we can **chop the pattern
into pieces**, judge each piece separately, and then ask "how many pieces agreed?"
This sounds like a small change; it is not. Judging shorter pieces means each stored
pattern tolerates a much wider range of speeds — so we need far fewer speed variants
stored, and the chip shrinks a lot.

The cost is sensitivity: chopping throws away some of the evidence, so you have to
listen for longer.

| Pieces | Speeds to store | Chip size | Time to find a satellite |
|---|---|---|---|
| 1 | 41 | 5.37 Mbit | 17 ms |
| 2 | 21 | 2.75 Mbit | 23 ms |
| 3 | 15 | 1.96 Mbit | 27 ms |
| 6 | 9 | 1.18 Mbit | 42 ms |
| 11 | 5 | 0.65 Mbit | 65 ms |

**An 8x smaller chip costs about 4x more time.** For GPS that is usually a good
trade: a receiver starting from cold is allowed to take a second or more to find
satellites, so time is cheap, while chip area is not.

## 8. The hard part, stated plainly

Because the evidence gap is only about 2% of the total count, the chip has to **count
very accurately**. It is not looking for a clear, obvious match; it is looking for a
slight statistical lean. We measured how accurate it must be: the memory needs to
distinguish differences of about **1 part in 100** of its total count.

That is demanding for this kind of circuit, and it is the main open question for
whoever builds the hardware. The good news is that the chopping trick from the
previous section relaxes it to about 1 part in 40 — so the same dial that shrinks the
chip also makes it easier to build. Time pays for both.

## 9. What this does not cover

- **Simulation only.** No circuit has been built. The "1 part in 100" figure is a
  requirement we are placing on a future counting circuit, not a measurement of one.
- **Clean conditions.** No interference, no signal bouncing off buildings, no
  automatic-gain-control dynamics.
- **Coarsest input setting only.** A slightly richer way of feeding the signal in is
  implemented and should recover a little of the gap, but we have not measured it
  end to end.
- **Finding, not tracking.** Once a satellite is found, a separate stage takes over
  to follow it precisely. This document is only about finding.
