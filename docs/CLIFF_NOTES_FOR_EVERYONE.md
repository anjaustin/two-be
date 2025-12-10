# The Path of Least Resistance
## Cliff Notes for Everyone

*No PhD required. If you can add numbers, you can understand this.*

---

## The One-Sentence Version

**We taught an AI to do perfect math—not by making it smarter, but by changing how we showed it the numbers.**

---

## The Problem (In Plain English)

You know how AI can write poetry, recognize faces, and beat humans at chess?

Here's something weird: **AI is terrible at basic arithmetic.**

Ask ChatGPT to add two large numbers. It'll probably get it wrong. This seems backwards—computers are supposed to be good at math!

We ran into this exact problem. We were building an AI to simulate an old computer chip (the 6502, from 1975). Our AI scored:
- **99.9%** on moving data around
- **3.1%** on addition

Three percent. On addition. A calculator from the dollar store does better.

---

## The Discovery

We spent months trying to fix this. More training data. Bigger AI. Fancier techniques.

Nothing worked.

Then we asked a different question:

> **"What if the AI can SEE the data, but can't see the PATTERN?"**

Think about how computers store numbers. The number 127 looks like this in binary:

```
127 = 01111111
```

And the number 128 looks like this:

```
128 = 10000000
```

**Every single digit changed.** To the AI, 127 and 128 look completely different—like "cat" and "xylophone."

But to us, 127 and 128 are neighbors. One apart. Almost the same.

**The AI couldn't see that they were neighbors because of how we wrote the numbers down.**

---

## The Solution

We changed how we wrote the numbers.

Instead of binary (0s and 1s), we used something like an abacus. On an abacus, you show a number by how many beads are pushed over:

```
5 = ●●●●●○○○○○  (5 beads pushed)
6 = ●●●●●●○○○○  (6 beads pushed)
```

See how 5 and 6 look almost the same? Just one bead different.

We did this for our AI. In our new system:

```
127 = ●●●●●●●●●●●●●●●○  |  ●●●●●●●○○○○○○○○○
128 = ○○○○○○○○○○○○○○○○  |  ●●●●●●●●○○○○○○○○
```

Now 127 and 128 only differ by a little bit. The AI can see they're neighbors.

---

## The Result

Same AI. Same training. Just different number format.

| Before | After |
|--------|-------|
| 3.1% accuracy | **100% accuracy** |
| 2.4 million parameters | 60,000 parameters |
| Never learned | Learned in 20 minutes |

**One hundred percent.** On 5 million test problems. Zero mistakes.

We didn't make the AI smarter. We made the problem *visible*.

---

## Why This Matters

### The Small Insight

Bad formatting hides patterns from AI. Good formatting reveals them.

### The Medium Insight

When people say "AI can't do X," they might be wrong. Maybe we just haven't found the right way to show AI the problem yet.

### The Big Insight

**This applies to everything.**

Biology is full of "invisible patterns"—things that look smooth but actually have hidden tipping points:
- Genes that are "off" until suddenly they're "on"
- Drugs that do nothing until suddenly they work (or cause harm)
- Cells that are healthy until suddenly they're cancerous

Current AI looks at biology with the equivalent of bad number formatting. It can see the data but can't see the patterns.

If we find the "abacus format" for biology, we might unlock the same kind of improvement—from "AI can't predict this" to "AI predicts this perfectly."

---

## The Metaphor

Imagine you're trying to walk to a destination, but you're blindfolded.

Someone is giving you directions: "You're getting warmer... warmer... colder... warmer..."

**Bad formatting** is like being in a maze where you can hear "warmer" but every step takes you somewhere random. The directions don't help because the path is invisible.

**Good formatting** is like being on a smooth downhill slope. "Warmer" always means "take a step forward." You just walk straight there.

The destination didn't move. The terrain changed.

> **We sculpted a landscape where the answer was downhill.**

---

## What We're NOT Saying

We're not saying:
- ❌ "We solved AI"
- ❌ "Scale doesn't matter"
- ❌ "This fixes everything"

We ARE saying:
- ✅ "When AI fails, check the formatting"
- ✅ "The right format can make impossible problems easy"
- ✅ "This principle applies to many domains"

---

## The Practical Takeaways

### If You Work with AI

Before throwing more data or bigger models at a problem, ask:
1. What patterns exist in this data?
2. Can the AI see those patterns in the current format?
3. What format would make the patterns obvious?

### If You Work in Biology/Medicine

The same principle applies:
1. Biological systems have hidden tipping points
2. Current data formats might hide them
3. Finding the right format could transform predictions

### If You're Just Curious

The next time someone says "AI can't do that," ask:
- "Is that a real limitation, or a formatting problem?"
- "What would we need to change to make it visible?"

---

## The Key Terms (Translated)

| Fancy Term | Plain English |
|------------|---------------|
| Representation | How you write down the data |
| Loss landscape | The terrain the AI walks on |
| Gradient descent | AI walking downhill toward answers |
| Soroban encoding | Abacus-style number format |
| Thermometer encoding | Format where bigger = more filled in |
| Structure visibility | Can the AI see the pattern? |

---

## Frequently Asked Questions

**Q: Why didn't anyone think of this before?**

A: Some people did, partially. But the field got distracted by "bigger is better." When making AI bigger sometimes works, you stop asking why it fails.

**Q: Does this mean we don't need big AI models?**

A: No. Big models help with some problems. But formatting helps with different problems. The best approach uses both.

**Q: Can AI learn the right format by itself?**

A: Sometimes partially. But finding the perfect format usually requires understanding the domain. That's why human expertise still matters.

**Q: What's next?**

A: We're looking at biology. Proteins, genes, drugs—all have hidden patterns. If we find the right formats, we might see similar breakthroughs.

---

## The One-Liner to Remember

> **AI isn't stupid. It's just sometimes looking at the data through foggy glasses. Clear the fog, and it sees perfectly.**

---

## Share This

If this made sense to you, share it. The more people understand this principle, the more problems we can solve.

The insight is simple: **formatting matters.**

The implications are huge: **"impossible" might just mean "not yet visible."**

---

*Written for humans, not journals.*
