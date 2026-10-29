# Baseline vs GRPO: GSM8K results

Generated 2026-10-29T02:10:11.948101+00:00.

## Evaluation settings

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Adapter | `outputs/grpo-qwen2.5-0.5b/final_adapter` |
| Split | `test` |
| Examples | 50 |
| Seed | 42 |
| Prompt version | `v1` |
| Max new tokens | 512 |
| Temperature | 0.0 |

Settings match between both runs: **yes**.

## Accuracy

| Stage | Correct | Accuracy | Format adherence |
| --- | ---: | ---: | ---: |
| Baseline (untrained) | 17/50 | 34.0% | 0.0% |
| GRPO adapter | 23/50 | 46.0% | 0.0% |
| Change | +6 | +12.0 pp | +0.0 pp |

Format adherence is the fraction of completions matching the exact `<think>...</think><answer>...</answer>` template.

## Extraction sensitivity

Strict scoring uses the extracted final answer (falling back to the last number). The lenient upper bound counts a run as correct if the gold value appears anywhere.

| Stage | Strict | Lenient upper bound |
| --- | ---: | ---: |
| Baseline (untrained) | 17/50 (34.0%) | 25/50 (50.0%) |
| GRPO adapter | 23/50 (46.0%) | 27/50 (54.0%) |
| Change | +12.0 pp | +4.0 pp |

The gap between these bounds shows how sensitive the result is to final-answer formatting. Use the strict number as the primary result.

## Paired outcomes

| Outcome | Count |
| --- | ---: |
| Improved | 8 |
| Regressed | 2 |
| Both correct | 15 |
| Both wrong | 25 |

McNemar exact two-sided p = 0.109. This is not statistically significant at 0.05.

## Side-by-side samples

### Example 1 (test index 60)

**Question:** A basket contains 25 oranges among which 1 is bad, 20% are unripe, 2 are sour and the rest are good. How many oranges are good?

**Gold answer:** `17`

**Baseline** — prediction `8` (wrong)

```text
Let's solve this step-by-step:

1) We know that there are 25 oranges in total.

2) Out of these, 1 orange is bad.

3) There are also 20% unripe oranges. To find out how many unripe oranges there are, we can calculate 20% of 25:
   (20/100) * 25 = 0.2 * 25 = 5

4) Now, let's add up all the good oranges:
   Good oranges = Bad oranges + Unripe oranges + Sour oranges
                  = 1 + 5 + 2
                  = 8

Therefore, there are 8 good oranges in the basket.
```

**GRPO** — prediction `17` (correct)

```text
Let's break down the information given:

1. The total number of oranges in the basket is 25.
2. 1 orange is bad.
3. 20% of the oranges are unripe.
4. There are 2 sour oranges.

First, let's calculate the number of unripe oranges:
\[ \text{Number of unripe oranges} = 20\% \times 25 = 0.20 \times 25 = 5 \]

Next, we know that there are 2 sour oranges. So, the remaining oranges must be good:
\[ \text{Number of good oranges} = 25 - (\text{Number of bad oranges} + \text{Number of unripe oranges} + \text{Number of sour oranges}) \]
\[ \text{Number of good oranges} = 25 - (1 + 5 + 2) = 25 - 8 = 17 \]

Therefore, the number of good oranges is 17.
```

### Example 2 (test index 457)

**Question:** While playing with her friends in their school playground, Katelyn saw 50 fairies flying above the nearby forest. After about twenty minutes, one of her friends saw half as many fairies as Katelyn saw come from the east and join the fairies that were there. Ten minutes later, 30 fairies flew away. How many fairies are remaining?

**Gold answer:** `45`

**Baseline** — prediction `0` (wrong)

```text
Let's break down the problem step by step:

1. Initially, Katelyn saw 50 fairies.

2. One of her friends saw half as many fairies as Katelyn did. So, she saw \( \frac{50}{2} = 25 \) fairies.

3. After ten more minutes, 30 fairies flew away.

4. To find out how many fairies remain, we need to subtract the number of fairies seen by both Katelyn and her friend from the initial count.

   - Fairies seen by Katelyn: 50
   - Fairies seen by her friend: 25
   - Fairies that flew away: 30

   Total fairies that flew away: \( 50 + 25 + 30 = 105 \)

5. Subtract the total fairies that flew away from the initial count:

   \( 50 - 105 = -55 \)

Since it's not possible for the number of fairies to be negative, it means all the fairies that flew away have been accounted for. Therefore, the final number of fairies remaining is 0.
```

**GRPO** — prediction `45` (correct)

```text
Let's break down the problem step by step:

1. Initially, Katelyn saw 50 fairies.

2. One of her friends saw half as many fairies as Katelyn did. So, she saw \( \frac{50}{2} = 25 \) fairies.

3. After ten more minutes, 30 fairies flew away.

4. To find out how many fairies remain, we need to subtract the number of fairies seen by both Katelyn and her friend plus those that flew away from the initial count.

So, the calculation is:
\[ 50 - 25 + (50 - 30) \]

Now let's do the math:
\[ 50 - 25 = 25 \]
\[ 25 + 50 - 30 = 45 \]

Therefore, the final number of fairies remaining is 45.
```

### Example 3 (test index 618)

**Question:** The school auditorium has 4 rows of seats. There are 18 seats in each row. One-fourth of the seats were occupied by the administrators. One-third of the remaining seats were occupied by the parents and the rest were occupied by the students. How many students were there in the school auditorium?

**Gold answer:** `36`

**Baseline** — prediction `42` (wrong)

```text
Let's solve this step-by-step:

1) First, we need to find out how many seats were occupied by the administrators. Since one-fourth of the seats were occupied by the administrators, we can calculate that as follows:
   Admins = (1/4) * Total Seats
   Admins = (1/4) * 4 * 18
   Admins = 9

2) Now, let's subtract the number of seats occupied by the administrators from the total number of seats to find out how many seats remain:
   Remaining Seats = Total Seats - Admins
   Remaining Seats = 4 * 18 - 9
   Remaining Seats = 72 - 9
   Remaining Seats = 63

3) Next, we'll find out how many seats were occupied by the parents by calculating one-third of the remaining seats:
   Parents = (1/3) * Remaining Seats
   Parents = (1/3) * 63
   Parents = 21

4) Finally, we'll subtract the number of seats occupied by the parents from the remaining seats to find out how many seats remained for the students:
[...]
```

**GRPO** — prediction `36` (correct)

```text
Let's break down the problem step by step:

1. **Calculate the total number of seats**: 
   - Each row has 18 seats, and there are 4 rows.
   - Total seats = \( 18 \times 4 = 72 \) seats.

2. **Determine how many seats were occupied by the administrators**:
   - One-fourth of the seats were occupied by the administrators.
   - Number of seats occupied by administrators = \( \frac{1}{4} \times 72 = 18 \) seats.

3. **Calculate the remaining seats after administrators**:
   - Remaining seats = Total seats - Seats occupied by administrators
   - Remaining seats = \( 72 - 18 = 54 \) seats.

4. **Determine how many seats were occupied by the parents**:
   - One-third of the remaining seats were occupied by the parents.
   - Number of seats occupied by parents = \( \frac{1}{3} \times 54 = 18 \) seats.

5. **Calculate the remaining seats after parents**:
   - Remaining seats after parents = Rem
[...]
```
