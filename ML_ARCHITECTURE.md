# ML Architecture Reference: Symmetric N-Tuple TD Learning

This document describes the machine learning system used in this 2048 bot project.
It is written as a reusable reference so you can explain this approach to any AI assistant
and have them replicate or adapt it for similar game-playing agents.

---

## 1. The Core Idea: Temporal Difference Learning (TD(0))

Instead of hard-coding heuristics ("keep the big tile in the corner"), the bot learns
to evaluate board states by playing millions of games against itself.

The key equation is the **TD(0) update rule**:

    V(s) ← V(s) + α * [r + V(s') - V(s)]

Where:
- V(s) = the learned value of the current board state (afterstate)
- V(s') = the learned value of the next board state after the environment responds
- r = the immediate reward (points scored from merging tiles)
- α = learning rate

The agent plays a move, observes the reward and next state, and adjusts its estimate
of the current state's value. Over millions of games, these values converge to an
accurate assessment of how "good" any board position is.

### Why "Afterstate"?

In 2048, after the player slides tiles, the board reaches a deterministic "afterstate."
Then the environment randomly places a new tile (2 or 4). We evaluate and learn on
the *afterstate* (after the player's move, before the random tile) because that is
the part the agent controls. This is more sample-efficient than learning on the full
state which includes randomness the agent can't influence.

---

## 2. State Representation: N-Tuple Networks

The 4x4 board has 16 cells, each holding a tile value from 0 to 15 (as log2).
Representing the full board as a single lookup table would require 16^16 entries,
which is impossible.

### Solution: Break the board into small overlapping shapes ("tuples")

Instead of one giant table, we define small groups of 4-6 cells on the board.
Each group is called a "tuple" or "feature." For each tuple, we maintain a lookup
table that maps every possible combination of tile values in those cells to a
learned weight.

**Example with 6-tuples (our architecture):**
A 2x3 rectangle on the board has 6 cells. Each cell can hold values 0-15 (log2 of
the tile). So the lookup table has 16^6 = 16,777,216 entries.

We define three types of 6-tuple shapes:
1. **Rectangles (2x3):** 6 positions covering rows/cols in a 2x3 block
2. **L-Shape 1:** 4 cells in a row + 2 cells below-left
3. **L-Shape 2:** 4 cells in a row + 2 cells below-center

Each shape type has its own weight table (so 3 tables total).

### The Evaluation Function

To evaluate a board, we:
1. Extract every tuple's cell values from the board
2. Pack them into a single integer index: (a << 20) | (b << 16) | (c << 12) | (d << 8) | (e << 4) | f
3. Look up the weight at that index in the corresponding table
4. Sum all the weights across all tuples

The total sum is the board's estimated value.

---

## 3. Symmetric Sampling (8-Way Symmetry)

A 4x4 board has 8 symmetries: 4 rotations × 2 reflections. A pattern learned in
the top-left corner should apply equally to the bottom-right corner (rotated 180°).

Instead of hoping the agent eventually encounters all rotations, we **force it** by
evaluating every tuple across all 8 symmetries of the board simultaneously.

**Implementation:**
We precompute index mappings for all 8 board transformations at startup. For each
of the 3 tuple shapes, across 8 symmetries, we get:
- Rectangles: 6 positions × 8 symmetries = 48 feature extractions
- L-Shape 1: 3 positions × 8 symmetries = 24 feature extractions
- L-Shape 2: 3 positions × 8 symmetries = 24 feature extractions
- **Total: 96 feature extractions per board evaluation**

**Critical detail for the learning rate:**
Since each weight gets updated 96 times per board (once per symmetric extraction),
the effective learning rate is amplified. To prevent divergence, divide alpha by 96:

    adj_alpha = alpha / 96.0

---

## 4. Performance Optimization with Numba

Pure Python is far too slow for this. We use **Numba** (`@njit` decorator) to
JIT-compile all hot-path functions to native machine code.

Key functions to JIT-compile:
- `evaluate_afterstate()` — the neural network forward pass
- `update_tables()` — the weight update (backprop equivalent)
- `get_best_move()` — the greedy action selector
- `play_game()` — the full game loop
- All board movement functions (`move_board`, `spawn_tile`)

### Parallelism

Use `@njit(nogil=True)` so Numba releases Python's GIL. Then use
`concurrent.futures.ThreadPoolExecutor` to run many games in parallel across
all CPU cores. Each thread has its own game but shares the same weight tables.

**Note:** Shared weight tables with unsynchronized writes is intentional.
The slight race conditions act as natural noise that actually helps exploration
and prevents overfitting. This is a well-known technique in RL called
"Hogwild" training.

---

## 5. The Training Loop (Pseudocode)

```
initialize weight_tables to all zeros (shape: [num_tuple_types, 16^6])
alpha = 0.0025  # initial learning rate

for each game:
    board = new_game()
    move, afterstate, score, value = get_best_move(board)
    
    while game is not over:
        next_board = spawn_random_tile(afterstate)
        next_move, next_afterstate, next_score, next_value = get_best_move(next_board)
        
        if no valid moves:
            # Game over: target value is 0
            td_error = 0 - evaluate(afterstate)
        else:
            # TD error: how wrong was our estimate?
            td_error = next_value - evaluate(afterstate)
        
        # Update weights for all 96 symmetric features
        for each symmetric feature index in afterstate:
            weight_tables[feature] += (alpha / 96) * td_error
        
        afterstate = next_afterstate
    
    # Decay alpha slowly over time
    alpha = max(0.0001, alpha * decay_factor)
```

---

## 6. The Agent (Playing in Real-Time)

At deployment, the trained weights are loaded from a file (weights.npy).
The agent uses **Expectimax search** (not Minimax, because the opponent is random):

- **Player nodes:** Try all 4 moves, pick the one with the highest expected value
- **Chance nodes:** Average over all possible random tile placements (2 with 90% prob, 4 with 10% prob)
- **Leaf nodes:** Evaluate using the trained N-tuple network

Dynamic depth adjustment:
- 3-ply when the board has many empty cells (fast, good enough)
- 4-ply when ≤6 empty cells
- 5-ply when ≤3 empty cells (critical decisions need deeper search)

---

## 7. Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Tuple size | 6 | Balances expressiveness vs memory (16^6 = 16.7M entries) |
| Num tuple types | 3 | Rectangles, L-shape 1, L-shape 2 |
| Symmetries | 8 | All rotations and reflections |
| Initial alpha | 0.0025 | Learning rate |
| Min alpha | 0.0001 | Floor to prevent stalling |
| Alpha divisor | 96 | = total symmetric features, prevents divergence |
| Weight dtype | float32 | Good enough precision, saves memory |

---

## 8. Why This Works So Well

1. **Massive pattern recognition:** 16.7 million states per shape means the agent
   memorizes the value of every possible local tile configuration it has ever seen.

2. **Symmetry exploitation:** 8x learning efficiency. A single game teaches the
   agent about patterns in all orientations simultaneously.

3. **No neural network overhead:** Unlike deep RL (DQN, PPO), there are no matrix
   multiplications, no backpropagation through layers, no GPU needed. It's just
   integer hashing + table lookups, making it extremely fast.

4. **Afterstate learning:** By learning on the deterministic part of the game,
   the agent doesn't waste capacity modeling randomness it can't control.

5. **Self-play convergence:** The agent improves its own evaluation function by
   playing against itself. Better evaluation → better moves → better training data
   → better evaluation. This positive feedback loop drives rapid improvement.

---

## 9. Adapting This to Other Games

This architecture works for any game where:
- The state space can be decomposed into small overlapping regions
- The game has symmetries (rotations, reflections)
- There is a clear reward signal
- The state transitions are either deterministic or have a small random component

**Examples:** Threes!, other puzzle games, simple board games.

For games with continuous state spaces or complex visual inputs, you would need
to replace the lookup tables with a neural network (deep RL), which is a
fundamentally different and more complex approach.
